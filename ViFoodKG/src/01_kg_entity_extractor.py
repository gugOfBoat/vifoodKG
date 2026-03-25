"""
Step 1A — Fetch all unique food_items from Supabase and save to JSON.
Simple: connect → read → deduplicate → save. That's it.

Usage:
  python src/01_kg_entity_extractor.py
"""

import json
import os
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_FILE = PROJECT_ROOT / "data" / "raw_unique_labels.json"


def main():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    client = create_client(url, key)
    print(f"✓ Connected to Supabase: {url}")

    # Fetch all rows with food_items, paginated
    all_items: list[list[str]] = []
    page, size = 0, 1000
    while True:
        resp = (
            client.table("image")
            .select("image_id, food_items")
            .not_.is_("food_items", "null")
            .range(page * size, (page + 1) * size - 1)
            .execute()
        )
        if not resp.data:
            break
        for row in resp.data:
            all_items.append(row["food_items"])
        print(f"  page {page}: {len(resp.data)} rows")
        if len(resp.data) < size:
            break
        page += 1

    # Flatten + deduplicate
    unique = set()
    for items in all_items:
        for lbl in items:
            cleaned = unicodedata.normalize("NFC", lbl).strip()
            if len(cleaned) >= 2:
                unique.add(cleaned)

    result = sorted(unique)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Done!")
    print(f"  Total rows: {sum(len(x) for x in all_items)}")
    print(f"  Unique labels: {len(result)}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
