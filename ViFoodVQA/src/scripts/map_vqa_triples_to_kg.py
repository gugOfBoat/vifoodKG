from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

PAGE_SIZE = 1000
BATCH_SIZE = 500


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_supabase_client():
    from supabase import create_client

    load_dotenv(find_dotenv(usecwd=True) or None)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY missing in .env")

    return create_client(url, key)


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")

    return data


def fetch_all_rows(
    client,
    table_name: str,
    select_cols: str,
    order_col: str,
    page_size: int = PAGE_SIZE,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0

    while True:
        resp = (
            client.table(table_name)
            .select(select_cols)
            .order(order_col)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break

        rows.extend(batch)
        print(f"Loaded {len(rows):,} rows from {table_name}...")

        if len(batch) < page_size:
            break

        start += page_size

    return rows


def parse_jsonish(value: Any) -> Any:
    """
    Accept:
    - Python list/dict (jsonb already decoded by Supabase client)
    - JSON string
    - None / empty string
    """
    if value is None:
        return []

    if isinstance(value, (list, dict)):
        return value

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []

    return []


def canonicalize_triple(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    Normalize a triple-like object to:
      {
        subject, relation, target, evidence, source_url
      }
    Ignore extra keys.
    """
    if not isinstance(item, dict):
        return None

    subject = norm_text(item.get("subject"))
    relation = norm_text(item.get("relation"))
    target = norm_text(item.get("target"))

    if not subject or not relation or not target:
        return None

    evidence = norm_text(item.get("evidence")) or None
    source_url = norm_text(item.get("source_url")) or None

    return {
        "subject": subject,
        "relation": relation,
        "target": target,
        "evidence": evidence,
        "source_url": source_url,
    }


def parse_triple_list(value: Any) -> list[dict[str, Any]]:
    data = parse_jsonish(value)
    if not isinstance(data, list):
        return []

    triples: list[dict[str, Any]] = []
    for item in data:
        triple = canonicalize_triple(item)
        if triple is not None:
            triples.append(triple)
    return triples


def extract_retrieved_from_food_items(item: dict[str, Any]) -> list[str]:
    """
    Flexible parser for optional metadata embedded in triples_retrieved.
    Supports keys like:
      - food_item
      - anchor_food_item
      - retrieved_from
      - retrieved_from_food_items
      - food_items
    """
    if not isinstance(item, dict):
        return []

    candidates: list[str] = []

    scalar_keys = ["food_item", "anchor_food_item", "retrieved_from"]
    list_keys = ["retrieved_from_food_items", "food_items"]

    for key in scalar_keys:
        value = norm_text(item.get(key))
        if value:
            candidates.append(value)

    for key in list_keys:
        value = item.get(key)
        if isinstance(value, list):
            for v in value:
                text = norm_text(v)
                if text:
                    candidates.append(text)

    # dedupe but keep order
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def chunks(seq: list[dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_triples_and_mapping(
    vqa_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns:
      1) unique triples for kg_triple_catalog
      2) mapping snapshot rows keyed by (vqa_id, triple_key)

    Mapping row shape before resolving triple_id:
      {
        vqa_id,
        subject,
        relation,
        target,
        is_used,
        is_retrieved,
        used_order,
        retrieval_rank,
        retrieved_from_food_items,
      }
    """
    triple_catalog: dict[tuple[str, str, str], dict[str, Any]] = {}
    mapping: dict[tuple[int, tuple[str, str, str]], dict[str, Any]] = {}

    for row in vqa_rows:
        vqa_id = row.get("vqa_id")
        if vqa_id is None:
            continue
        vqa_id = int(vqa_id)

        used_raw = parse_jsonish(row.get("triples_used"))
        retrieved_raw = parse_jsonish(row.get("triples_retrieved"))

        used_list = used_raw if isinstance(used_raw, list) else []
        retrieved_list = retrieved_raw if isinstance(retrieved_raw, list) else []

        # 1) retrieved triples
        for idx, item in enumerate(retrieved_list):
            triple = canonicalize_triple(item)
            if triple is None:
                continue

            key = (triple["subject"], triple["relation"], triple["target"])

            # merge triple metadata
            if key not in triple_catalog:
                triple_catalog[key] = dict(triple)
            else:
                existing = triple_catalog[key]
                if not existing.get("evidence") and triple.get("evidence"):
                    existing["evidence"] = triple["evidence"]
                if not existing.get("source_url") and triple.get("source_url"):
                    existing["source_url"] = triple["source_url"]

            map_key = (vqa_id, key)
            retrieved_from_items = extract_retrieved_from_food_items(item)

            if map_key not in mapping:
                mapping[map_key] = {
                    "vqa_id": vqa_id,
                    "subject": triple["subject"],
                    "relation": triple["relation"],
                    "target": triple["target"],
                    "is_used": False,
                    "is_retrieved": True,
                    "used_order": None,
                    "retrieval_rank": idx,
                    "retrieved_from_food_items": retrieved_from_items,
                }
            else:
                entry = mapping[map_key]
                entry["is_retrieved"] = True

                if entry["retrieval_rank"] is None or idx < entry["retrieval_rank"]:
                    entry["retrieval_rank"] = idx

                merged_items = entry.get("retrieved_from_food_items") or []
                seen = set(merged_items)
                for food in retrieved_from_items:
                    if food not in seen:
                        seen.add(food)
                        merged_items.append(food)
                entry["retrieved_from_food_items"] = merged_items

        # 2) used triples
        for idx, item in enumerate(used_list):
            triple = canonicalize_triple(item)
            if triple is None:
                continue

            key = (triple["subject"], triple["relation"], triple["target"])

            # merge triple metadata
            if key not in triple_catalog:
                triple_catalog[key] = dict(triple)
            else:
                existing = triple_catalog[key]
                if not existing.get("evidence") and triple.get("evidence"):
                    existing["evidence"] = triple["evidence"]
                if not existing.get("source_url") and triple.get("source_url"):
                    existing["source_url"] = triple["source_url"]

            map_key = (vqa_id, key)

            if map_key not in mapping:
                mapping[map_key] = {
                    "vqa_id": vqa_id,
                    "subject": triple["subject"],
                    "relation": triple["relation"],
                    "target": triple["target"],
                    "is_used": True,
                    "is_retrieved": False,
                    "used_order": idx,
                    "retrieval_rank": None,
                    "retrieved_from_food_items": [],
                }
            else:
                entry = mapping[map_key]
                entry["is_used"] = True
                if entry["used_order"] is None or idx < entry["used_order"]:
                    entry["used_order"] = idx

    return list(triple_catalog.values()), list(mapping.values())


def upsert_triples_to_catalog(
    client,
    triples: list[dict[str, Any]],
    table_name: str = "kg_triple_catalog",
    batch_size: int = BATCH_SIZE,
) -> None:
    if not triples:
        print("No triples to upsert into kg_triple_catalog.")
        return

    done = 0
    for batch in chunks(triples, batch_size):
        client.table(table_name).upsert(
            batch,
            on_conflict="subject,relation,target",
        ).execute()
        done += len(batch)
        print(f"Upserted triples: {done:,}/{len(triples):,}")


def build_kg_lookup(
    client,
    table_name: str = "kg_triple_catalog",
) -> dict[tuple[str, str, str], int]:
    kg_rows = fetch_all_rows(
        client=client,
        table_name=table_name,
        select_cols="triple_id,subject,relation,target",
        order_col="triple_id",
    )

    lookup: dict[tuple[str, str, str], int] = {}
    for row in kg_rows:
        subject = norm_text(row.get("subject"))
        relation = norm_text(row.get("relation"))
        target = norm_text(row.get("target"))
        triple_id = row.get("triple_id")

        if subject and relation and target and triple_id is not None:
            lookup[(subject, relation, target)] = int(triple_id)

    return lookup


def resolve_mapping_rows(
    mapping_candidates: list[dict[str, Any]],
    kg_lookup: dict[tuple[str, str, str], int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    timestamp = now_iso()

    for item in mapping_candidates:
        key = (
            item["subject"],
            item["relation"],
            item["target"],
        )
        triple_id = kg_lookup.get(key)

        if triple_id is None:
            missing.append(item)
            continue

        is_used = bool(item["is_used"])
        is_retrieved = bool(item["is_retrieved"])

        resolved.append(
            {
                "vqa_id": item["vqa_id"],
                "triple_id": triple_id,
                "is_used": is_used,
                "is_retrieved": is_retrieved,
                # A triple is considered active for a VQA only when it belongs
                # to the final triples_used set for that question.
                "is_active_for_vqa": is_used,
                # This script is only a structural backfill from raw JSON fields.
                # It must not mark rows as human-reviewed.
                "triple_review_status": None,
                "triple_review_note": None,
                "used_order": item.get("used_order"),
                "retrieval_rank": item.get("retrieval_rank"),
                "retrieved_from_food_items": item.get("retrieved_from_food_items") or [],
                "replaced_by_triple_id": None,
                "reviewed_from_page": None,
                "reviewed_at": None,
                "updated_at": timestamp,
            }
        )

    return resolved, missing


def upsert_mapping_rows(
    client,
    mapping_rows: list[dict[str, Any]],
    table_name: str = "vqa_kg_triple_map",
    batch_size: int = BATCH_SIZE,
) -> None:
    if not mapping_rows:
        print("No mapping rows to upsert.")
        return

    done = 0
    for batch in chunks(mapping_rows, batch_size):
        client.table(table_name).upsert(
            batch,
            on_conflict="vqa_id,triple_id",
        ).execute()
        done += len(batch)
        print(f"Upserted mappings: {done:,}/{len(mapping_rows):,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize vqa.triples_used + vqa.triples_retrieved into kg_triple_catalog and vqa_kg_triple_map"
    )
    parser.add_argument(
        "--input-json",
        default="",
        help="Optional local JSON file with VQA rows for testing",
    )
    parser.add_argument(
        "--vqa-table",
        default="vqa",
        help="VQA table name",
    )
    parser.add_argument(
        "--kg-table",
        default="kg_triple_catalog",
        help="KG catalog table name",
    )
    parser.add_argument(
        "--map-table",
        default="vqa_kg_triple_map",
        help="Mapping table name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for upsert, default={BATCH_SIZE}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print stats only, do not write to Supabase",
    )
    args = parser.parse_args()

    client = make_supabase_client()

    if args.input_json:
        vqa_rows = load_json_rows(Path(args.input_json))
        print(f"Loaded {len(vqa_rows):,} VQA rows from local JSON")
    else:
        vqa_rows = fetch_all_rows(
            client=client,
            table_name=args.vqa_table,
            select_cols="vqa_id,triples_used,triples_retrieved",
            order_col="vqa_id",
        )
        print(f"Loaded {len(vqa_rows):,} VQA rows from Supabase")

    triples, mapping_candidates = build_triples_and_mapping(vqa_rows)

    used_only = sum(
        1 for x in mapping_candidates if x["is_used"] and not x["is_retrieved"]
    )
    retrieved_only = sum(
        1 for x in mapping_candidates if x["is_retrieved"] and not x["is_used"]
    )
    both = sum(
        1 for x in mapping_candidates if x["is_retrieved"] and x["is_used"]
    )

    print(f"Unique triples found: {len(triples):,}")
    print(f"Candidate mappings: {len(mapping_candidates):,}")
    print(f"  used only      : {used_only:,}")
    print(f"  retrieved only : {retrieved_only:,}")
    print(f"  both           : {both:,}")

    if args.dry_run:
        print("Dry-run enabled. No data written.")
        print("Sample triples:")
        for item in triples[:5]:
            print(item)
        print("Sample mapping candidates:")
        for item in mapping_candidates[:5]:
            print(item)
        return

    upsert_triples_to_catalog(
        client=client,
        triples=triples,
        table_name=args.kg_table,
        batch_size=args.batch_size,
    )

    kg_lookup = build_kg_lookup(client=client, table_name=args.kg_table)

    mapping_rows, missing = resolve_mapping_rows(
        mapping_candidates=mapping_candidates,
        kg_lookup=kg_lookup,
    )

    print(f"Resolved mappings  : {len(mapping_rows):,}")
    print(f"Unresolved mappings: {len(missing):,}")

    if missing:
        print("First 10 unresolved mappings:")
        for item in missing[:10]:
            print(item)

    upsert_mapping_rows(
        client=client,
        mapping_rows=mapping_rows,
        table_name=args.map_table,
        batch_size=args.batch_size,
    )

    print("Done.")


if __name__ == "__main__":
    main()