from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


BATCH_SIZE = 500


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def chunks(seq: list[dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def make_neo4j_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not password:
        raise RuntimeError("Missing NEO4J_URI / NEO4J_PASSWORD in .env")

    return GraphDatabase.driver(uri, auth=(user, password))


def make_supabase_client():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_KEY in .env")

    return create_client(url, key)


def fetch_triples_from_neo4j(limit: int | None = None) -> list[dict[str, Any]]:
    query = """
    MATCH (s)-[r]->(t)
    WHERE s.name IS NOT NULL
      AND t.name IS NOT NULL
    RETURN
      s.name AS subject,
      type(r) AS relation,
      t.name AS target,
      coalesce(r.evidence, '') AS evidence,
      coalesce(r.source_url, '') AS source_url
    ORDER BY subject, relation, target
    """

    if limit is not None and limit > 0:
        query += "\nLIMIT $limit"
        params = {"limit": limit}
    else:
        params = {}

    driver = make_neo4j_driver()
    try:
        with driver.session() as session:
            rows = session.run(query, **params).data()
    finally:
        driver.close()

    return rows


def dedup_triples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate by (subject, relation, target).
    Keep the first non-empty evidence/source_url encountered.
    Do NOT include is_checked / is_drop in payload:
    - inserts will use DB defaults (false)
    - updates won't overwrite existing review states
    """
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    now_iso = datetime.now(timezone.utc).isoformat()

    for row in rows:
        subject = norm_text(row.get("subject"))
        relation = norm_text(row.get("relation"))
        target = norm_text(row.get("target"))
        evidence = norm_text(row.get("evidence")) or None
        source_url = norm_text(row.get("source_url")) or None

        if not subject or not relation or not target:
            continue

        key = (subject, relation, target)

        if key not in merged:
            merged[key] = {
                "subject": subject,
                "relation": relation,
                "target": target,
                "evidence": evidence,
                "source_url": source_url,
                "updated_at": now_iso,
            }
        else:
            if not merged[key].get("evidence") and evidence:
                merged[key]["evidence"] = evidence
            if not merged[key].get("source_url") and source_url:
                merged[key]["source_url"] = source_url

    return list(merged.values())


def truncate_table_if_needed(supabase, table_name: str) -> None:
    """
    Supabase Python client không có truncate tiện như SQL trực tiếp.
    Với table đang trống thì không cần dùng hàm này.
    Nếu muốn clear dữ liệu, tốt nhất chạy SQL:
      truncate table public.kg_triple_catalog restart identity;
    """
    raise NotImplementedError(
        "Please truncate with SQL in Supabase SQL editor if needed."
    )


def upsert_to_supabase(
    triples: list[dict[str, Any]],
    table_name: str,
    batch_size: int,
    dry_run: bool = False,
) -> None:
    if not triples:
        print("No triples to upsert.")
        return

    print(f"Prepared {len(triples)} unique triples for upsert")

    if dry_run:
        print("Dry-run enabled. No data written to Supabase.")
        print("Sample payload:")
        for item in triples[:5]:
            print(item)
        return

    supabase = make_supabase_client()

    done = 0
    for batch in chunks(triples, batch_size):
        supabase.table(table_name).upsert(
            batch,
            on_conflict="subject,relation,target",
        ).execute()

        done += len(batch)
        print(f"Upserted {done}/{len(triples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch triples from Neo4j Aura and upsert into Supabase"
    )
    parser.add_argument(
        "--table",
        default="kg_triple_catalog",
        help="Supabase table name (default: kg_triple_catalog)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit number of relationships fetched from Neo4j",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for Supabase upsert (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch and print sample payload, do not write to Supabase",
    )

    args = parser.parse_args()

    load_dotenv()

    print("Fetching triples from Neo4j...")
    rows = fetch_triples_from_neo4j(limit=args.limit)
    print(f"Fetched {len(rows)} relationship rows from Neo4j")

    triples = dedup_triples(rows)

    print("Upserting to Supabase...")
    upsert_to_supabase(
        triples=triples,
        table_name=args.table,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    print("Done.")


if __name__ == "__main__":
    main()