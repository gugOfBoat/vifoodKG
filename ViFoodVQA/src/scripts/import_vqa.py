"""
Import generated VQA JSON into Supabase table `vqa`.

Expected input file shape (from 05_generate_vqa.py):
[
  {
    "image_id": "image000001",
    "qtype": "ingredients",
    "question_vi": "...",
    "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "answer": "A",
    "rationale_vi": "...",
    "triples_used": [{"subject": "...", "relation": "...", "target": "..."}]
  }
]

Usage:
  python src/upsert_supabase/08_import_vqa.py
  python src/upsert_supabase/08_import_vqa.py --input data/vqa/generated_vqa.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_INPUT = PROJECT_ROOT / "data" / "vqa" / "generated_vqa.json"
PAGE_SIZE = 1000
BATCH_SIZE = 500


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def make_supabase_client():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY missing in .env")
    return create_client(url, key)


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")
    return data


def fetch_all_image_ids(client) -> set[str]:
    result: set[str] = set()
    start = 0
    while True:
        resp = (
            client.table("image")
            .select("image_id")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        result.update(norm_text(row.get("image_id")) for row in rows if norm_text(row.get("image_id")))
        if len(rows) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return result


def shrink_triples(triples: Any) -> list[dict[str, str]]:
    if not isinstance(triples, list):
        return []
    shrunk = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        subject = norm_text(t.get("subject"))
        relation = norm_text(t.get("relation"))
        target = norm_text(t.get("target"))
        if not subject or not relation or not target:
            continue
        shrunk.append({
            "subject": subject,
            "relation": relation,
            "target": target,
        })
    return shrunk


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
    image_id = norm_text(sample.get("image_id"))
    qtype = norm_text(sample.get("qtype") or sample.get("question_type"))
    question = norm_text(sample.get("question") or sample.get("question_vi"))
    answer = norm_text(sample.get("answer")).upper()
    rationale = norm_text(sample.get("rationale") or sample.get("rationale_vi")) or None
    choices = sample.get("choices") or {}

    if not image_id or not qtype or not question:
        return None
    if answer not in {"A", "B", "C", "D"}:
        return None

    choice_a = norm_text(choices.get("A"))
    choice_b = norm_text(choices.get("B"))
    choice_c = norm_text(choices.get("C"))
    choice_d = norm_text(choices.get("D"))
    if not all([choice_a, choice_b, choice_c, choice_d]):
        return None

    return {
        "image_id": image_id,
        "qtype": qtype,
        "question": question,
        "choice_a": choice_a,
        "choice_b": choice_b,
        "choice_c": choice_c,
        "choice_d": choice_d,
        "answer": answer,
        "rationale": rationale,
        "triples_used": shrink_triples(sample.get("triples_used")),
        "is_checked": False,
        "is_drop": False,
    }


def chunks(seq: list[dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import VQA JSON into Supabase")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to generated_vqa.json")
    args = parser.parse_args()

    records = load_json(Path(args.input))
    supabase = make_supabase_client()
    valid_image_ids = fetch_all_image_ids(supabase)
    print(f"Loaded {len(valid_image_ids)} image IDs from Supabase")

    normalized: list[dict[str, Any]] = []
    skipped = 0

    for raw in records:
        sample = normalize_sample(raw)
        if not sample:
            skipped += 1
            continue
        if sample["image_id"] not in valid_image_ids:
            skipped += 1
            continue
        normalized.append(sample)

    print(f"Prepared {len(normalized)} VQA samples for import")
    print(f"Skipped {skipped} invalid/orphan samples")

    imported = 0
    for batch in chunks(normalized, BATCH_SIZE):
        supabase.table("vqa").upsert(
            batch,
            on_conflict="image_id,qtype,question",
        ).execute()
        imported += len(batch)
        print(f"  Upserted {imported}/{len(normalized)}")

    print("Done.")


if __name__ == "__main__":
    main()
