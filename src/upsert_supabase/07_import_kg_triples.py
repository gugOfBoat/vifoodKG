"""
Import extracted KG triples into Supabase table `kg_triple_catalog`.

Expected input file shape (from 03_kg_triple_extractor.py):
[
  {
    "dish": "Bánh Xèo",
    "triples": [
      {
        "subject": "Bánh Xèo",
        "relation": "hasIngredient",
        "target": "Tôm",
        "source_url": "https://...",
        "evidence": "..."
      }
    ]
  }
]

Usage:
  python src/06_import_kg_triples.py
  python src/06_import_kg_triples.py --input data/triples/extracted_triples.json
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

DEFAULT_INPUT = PROJECT_ROOT / "data" / "triples" / "extracted_triples.json"
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


def flatten_unique_triples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}

    for record in records:
        triples = record.get("triples") or []
        if not isinstance(triples, list):
            continue
        for t in triples:
            if not isinstance(t, dict):
                continue
            subject = norm_text(t.get("subject"))
            relation = norm_text(t.get("relation"))
            target = norm_text(t.get("target"))
            if not subject or not relation or not target:
                continue

            key = (subject, relation, target)
            evidence = norm_text(t.get("evidence")) or None
            source_url = norm_text(t.get("source_url")) or None

            if key not in dedup:
                dedup[key] = {
                    "subject": subject,
                    "relation": relation,
                    "target": target,
                    "evidence": evidence,
                    "source_url": source_url,
                }
            else:
                existing = dedup[key]
                if not existing.get("evidence") and evidence:
                    existing["evidence"] = evidence
                if not existing.get("source_url") and source_url:
                    existing["source_url"] = source_url

    return list(dedup.values())


def chunks(seq: list[dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import extracted triples into Supabase")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to extracted_triples.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = load_json(input_path)
    triples = flatten_unique_triples(records)
    print(f"Loaded {len(records)} dish records")
    print(f"Flattened to {len(triples)} unique triples")

    supabase = make_supabase_client()

    imported = 0
    for batch in chunks(triples, BATCH_SIZE):
        supabase.table("kg_triple_catalog").upsert(
            batch,
            on_conflict="subject,relation,target",
        ).execute()
        imported += len(batch)
        print(f"  Upserted {imported}/{len(triples)}")

    print("Done.")


if __name__ == "__main__":
    main()
