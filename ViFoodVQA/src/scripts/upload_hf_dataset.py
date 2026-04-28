"""
Upload ViFoodVQA dataset to HuggingFace Hub.

Uses the `datasets` library to properly embed images into Parquet files,
so the HF dataset viewer can render image previews directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value

REPO_ID = "hoangphann/ViFoodVQA"
HF_DIR = Path("hf_dataset")
DATA_DIR = HF_DIR / "data"

SPLITS = ["train", "validation", "test"]

TRIPLE_KEYS = ("target", "subject", "relation")


def load_split(split: str) -> Dataset:
    jsonl_path = DATA_DIR / f"{split}.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing split file: {jsonl_path}")

    # Read JSONL and normalize triples_used to ensure consistent struct keys.
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            # Resolve relative image path to absolute for Image() encoding.
            img = row.get("image")
            if img and isinstance(img, str):
                row["image"] = str((HF_DIR / img).resolve())

            # Ensure every triple has exactly {target, subject, relation}.
            triples = row.get("triples_used") or []
            row["triples_used"] = [
                {k: t.get(k, "") for k in TRIPLE_KEYS} for t in triples
            ]

            rows.append(row)

    features = Features(
        {
            "vqa_id": Value("int64"),
            "image_id": Value("string"),
            "image": Image(),
            "qtype": Value("string"),
            "question": Value("string"),
            "choices": {
                "A": Value("string"),
                "B": Value("string"),
                "C": Value("string"),
                "D": Value("string"),
            },
            "answer": Value("string"),
            "rationale": Value("string"),
            "triples_used": [
                {
                    "target": Value("string"),
                    "subject": Value("string"),
                    "relation": Value("string"),
                }
            ],
        }
    )

    return Dataset.from_list(rows, features=features)


def main() -> None:
    splits: dict[str, Dataset] = {}

    for split in SPLITS:
        print(f"Loading {split}...")
        splits[split] = load_split(split)
        print(f"  {split}: {len(splits[split]):,} rows")

    dataset_dict = DatasetDict(splits)

    print(f"\nPushing to {REPO_ID}...")
    dataset_dict.push_to_hub(
        REPO_ID,
        private=True,
        commit_message="Upload ViFoodVQA dataset with embedded images",
    )

    print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()