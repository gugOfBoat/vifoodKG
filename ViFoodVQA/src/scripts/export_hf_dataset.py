"""
Export ViFoodVQA records from Supabase into a local Hugging Face dataset layout.

The script fetches rows from the Supabase `vqa` table, joins image metadata,
applies the current split-aware quality policy, normalizes each valid sample,
and writes JSONL files under `hf_dataset/data/`. It can also download referenced
images into `hf_dataset/images/` and store relative image paths in the exported
records.

Quality policy:
- test rows must be human-checked, not dropped, and marked KEEP.
- train and validation rows may be unverified, but must not be dropped.

Typical usage:
    python src/scripts/export_hf_dataset.py --export-by-split --download-images
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

PAGE_SIZE = 1000
REQUEST_TIMEOUT = 30

DEFAULT_HF_DIR = Path("hf_dataset")


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_image_id(image_id: str) -> str:
    text = norm(image_id)
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("?", "_")
        .replace("&", "_")
        .replace("=", "_")
        .replace(" ", "_")
    )


def make_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in .env")

    return create_client(url, key)


def parse_jsonish(value: Any) -> Any:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Split / quality policy
# ---------------------------------------------------------------------

def normalize_split(value: Any) -> str:
    split = norm(value).lower()

    if split in {"train", "training"}:
        return "train"

    # Supabase đang lưu validation split là "validate"
    if split in {"validate", "val", "valid", "validation", "dev"}:
        return "validation"

    if split in {"test", "testing"}:
        return "test"

    return "unknown"

def should_keep_row_by_split_policy(row: dict[str, Any]) -> bool:
    """
    Current ViFoodVQA export policy:

    - test:
        require human verification:
        is_checked = true
        is_drop = false
        verify_decision = KEEP

    - train / validation:
        allow generated but unverified samples:
        is_drop = false
        no need is_checked = true
        no need verify_decision = KEEP
    """
    split = normalize_split(row.get("split"))

    is_checked = row.get("is_checked") is True
    is_drop = row.get("is_drop") is True
    verify_decision = norm(row.get("verify_decision")).upper()

    if is_drop:
        return False

    if split == "test":
        return is_checked and verify_decision == "KEEP"

    if split in {"train", "validation"}:
        return True

    return False





# ---------------------------------------------------------------------
# Image source detection / download
# ---------------------------------------------------------------------

def detect_image_source_type(image_url: str) -> str:
    url = norm(image_url).lower()

    if not url:
        return "unknown"

    if "supabase.co/storage/v1/object" in url:
        return "supabase_storage"

    if "supabase.co" in url and "/storage/" in url:
        return "supabase_storage"

    if url.startswith("http://") or url.startswith("https://"):
        return "external_cdn"

    return "unknown"


def should_download_image(source_type: str, download_source: str) -> bool:
    if download_source == "all":
        return source_type in {"supabase_storage", "external_cdn", "unknown"}

    if download_source == "supabase-only":
        return source_type == "supabase_storage"

    if download_source == "external-only":
        return source_type == "external_cdn"

    return False


def guess_ext(image_url: str, content_type: str | None = None) -> str:
    parsed = urlparse(image_url)
    suffix = Path(parsed.path).suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix

    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        ext = mimetypes.guess_extension(mime)

        if ext in {".jpg", ".jpeg", ".png", ".webp"}:
            return ext

        if mime == "image/jpeg":
            return ".jpg"

    return ".jpg"


def find_existing_image(image_id: str, image_dir: Path, hf_dir: Path) -> str | None:
    """
    Fast resume mechanism:
    If image already exists with common extension, return relative path
    without requesting the URL again.
    """
    safe_id = safe_image_id(image_id)

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        existing_path = image_dir / f"{safe_id}{ext}"
        if existing_path.exists():
            return str(existing_path.relative_to(hf_dir)).replace("\\", "/")

    return None


def download_image(
    *,
    image_url: str,
    image_id: str,
    image_dir: Path,
    hf_dir: Path,
    overwrite: bool = False,
    retries: int = 2,
    sleep_seconds: float = 1.0,
) -> str | None:
    """
    Download image_url to image_dir.

    Returns:
        Relative path from hf_dir, e.g. images/image_000001.jpg
        or None if failed.
    """
    image_url = norm(image_url)
    image_id = safe_image_id(image_id)

    if not image_url or not image_id:
        return None

    image_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = find_existing_image(image_id, image_dir, hf_dir)
        if existing:
            return existing

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                image_url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    )
                },
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")

            if content_type and not content_type.lower().startswith("image/"):
                print(
                    f"[WARN] URL is not image: image_id={image_id}, "
                    f"content-type={content_type}"
                )
                return None

            ext = guess_ext(image_url, content_type)
            image_path = image_dir / f"{image_id}{ext}"

            if image_path.exists() and not overwrite:
                return str(image_path.relative_to(hf_dir)).replace("\\", "/")

            image_path.write_bytes(resp.content)

            return str(image_path.relative_to(hf_dir)).replace("\\", "/")

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds)

    print(
        f"[WARN] Failed to download image_id={image_id}, "
        f"url={image_url}, error={last_error}"
    )
    return None


# ---------------------------------------------------------------------
# Supabase fetch
# ---------------------------------------------------------------------

def build_select_query() -> str:
    return """
        vqa_id,
        image_id,
        split,
        is_checked,
        is_drop,
        qtype,
        question,
        choice_a,
        choice_b,
        choice_c,
        choice_d,
        answer,
        rationale,
        triples_used,
        verify_decision,
        image:image_id (
            image_url,
            is_drop
        )
    """


def fetch_candidate_rows(client) -> list[dict[str, Any]]:
    """
    Fetch candidates from Supabase.

    Important:
    Do NOT globally filter is_checked = true here, because currently only
    test split is fully checked. Split-aware filtering is handled in Python.
    """
    rows: list[dict[str, Any]] = []
    start = 0

    select_query = build_select_query()

    while True:
        resp = (
            client.table("vqa")
            .select(select_query)
            .eq("is_drop", False)
            .order("vqa_id")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )

        batch = resp.data or []

        if not batch:
            break

        rows.extend(batch)
        print(f"Fetched {len(rows):,} VQA rows...")

        if len(batch) < PAGE_SIZE:
            break

        start += PAGE_SIZE

    return rows


# ---------------------------------------------------------------------
# Row normalization
# ---------------------------------------------------------------------

def normalize_row(
    row: dict[str, Any],
    *,
    hf_dir: Path,
    image_dir: Path,
    download_images: bool,
    download_source: str,
    overwrite_images: bool,
) -> dict[str, Any] | None:
    if not should_keep_row_by_split_policy(row):
        return None

    image = row.get("image") or {}

    # Do not require image.is_checked = true, because train/val may not be checked yet.
    if image.get("is_drop") is True:
        return None

    image_id = norm(row.get("image_id"))
    image_url = norm(image.get("image_url"))

    if not image_id:
        return None

    image_source_type = detect_image_source_type(image_url)

    local_image_path: str | None = None

    if download_images and should_download_image(image_source_type, download_source):
        local_image_path = download_image(
            image_url=image_url,
            image_id=image_id,
            image_dir=image_dir,
            hf_dir=hf_dir,
            overwrite=overwrite_images,
        )

    choices = {
        "A": norm(row.get("choice_a")),
        "B": norm(row.get("choice_b")),
        "C": norm(row.get("choice_c")),
        "D": norm(row.get("choice_d")),
    }

    answer = norm(row.get("answer")).upper()
    question = norm(row.get("question"))
    qtype = norm(row.get("qtype"))

    if answer not in {"A", "B", "C", "D"}:
        return None

    if not all(choices.values()):
        return None

    if not question or not qtype:
        return None

    return {
        "vqa_id": row.get("vqa_id"),
        "image_id": image_id,
        "image": local_image_path,
        "qtype": qtype,
        "question": question,
        "choices": choices,
        "answer": answer,
        "rationale": row.get("rationale"),
        "triples_used": parse_jsonish(row.get("triples_used")),
    }


# ---------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------

def split_records(
    records: list[dict[str, Any]],
    split_map: dict[int, str],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
        "unknown": [],
    }

    for idx, item in enumerate(records):
        split = split_map.get(idx, "unknown")
        bucket = result.get(split, result["unknown"])
        bucket.append(item)

    return result


def write_outputs(
    *,
    records: list[dict[str, Any]],
    data_dir: Path,
    output_name: str,
    export_by_split: bool,
    split_map: dict[int, str] | None = None,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    all_path = data_dir / output_name
    write_jsonl(all_path, records)
    print(f"Saved: {all_path} ({len(records):,} rows)")

    if not export_by_split:
        return

    grouped = split_records(records, split_map or {})

    for split, split_rows in grouped.items():
        if not split_rows:
            continue

        output_path = data_dir / f"{split}.jsonl"
        write_jsonl(output_path, split_rows)
        print(f"Saved: {output_path} ({len(split_rows):,} rows)")


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------

def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: dict[str, int] = {}

    for item in records:
        value = norm(item.get(key)) or "unknown"
        counter[value] = counter.get(value, 0) + 1

    return counter


def print_counter(title: str, counter: dict[str, int]) -> None:
    print(f"\n{title}:")
    for key, value in sorted(counter.items()):
        print(f"  - {key}: {value:,}")


def print_stats(
    records: list[dict[str, Any]],
    skipped: int,
    split_map: dict[int, str],
) -> None:
    print("\n=== Export stats ===")
    print(f"Total exported records: {len(records):,}")
    print(f"Skipped rows: {skipped:,}")

    split_counter: dict[str, int] = {}
    for split in split_map.values():
        split_counter[split] = split_counter.get(split, 0) + 1
    print_counter("Split", split_counter)
    print_counter("QType", count_by(records, "qtype"))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ViFoodVQA from Supabase to Hugging Face dataset format."
    )

    parser.add_argument(
        "--hf-dir",
        default=str(DEFAULT_HF_DIR),
        help="Output Hugging Face dataset directory. Default: hf_dataset",
    )

    parser.add_argument(
        "--output-name",
        default="all.jsonl",
        help="Output JSONL filename under data/. Default: all.jsonl",
    )

    parser.add_argument(
        "--export-by-split",
        action="store_true",
        help="Write train.jsonl, validation.jsonl, test.jsonl.",
    )

    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download images from image_url into hf_dataset/images/.",
    )

    parser.add_argument(
        "--download-source",
        choices=["all", "supabase-only", "external-only"],
        default="all",
        help="Which image URL sources to download. Default: all",
    )

    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite existing images in hf_dataset/images/.",
    )



    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hf_dir = Path(args.hf_dir)
    data_dir = hf_dir / "data"
    image_dir = hf_dir / "images"

    client = make_supabase_client()

    raw_rows = fetch_candidate_rows(client)

    records: list[dict[str, Any]] = []
    split_map: dict[int, str] = {}  # index -> split (kept separate from output row)
    skipped = 0

    for row in raw_rows:
        item = normalize_row(
            row,
            hf_dir=hf_dir,
            image_dir=image_dir,
            download_images=args.download_images,
            download_source=args.download_source,
            overwrite_images=args.overwrite_images,
        )

        if item is None:
            skipped += 1
            continue

        split_map[len(records)] = normalize_split(row.get("split"))
        records.append(item)

    write_outputs(
        records=records,
        data_dir=data_dir,
        output_name=args.output_name,
        export_by_split=args.export_by_split,
        split_map=split_map,
    )

    print_stats(records, skipped, split_map)
    print("\nDone.")


if __name__ == "__main__":
    main()
