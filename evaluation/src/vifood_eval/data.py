from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import resolve_eval_path

REQUIRED_FIELDS = {
    "vqa_id",
    "image_id",
    "image",
    "qtype",
    "question",
    "choices",
    "answer",
    "triples_used",
}


@dataclass(frozen=True)
class VQASample:
    row: dict[str, Any]
    split: str
    data_dir: Path

    @property
    def vqa_id(self) -> int:
        return int(self.row["vqa_id"])

    @property
    def image_path(self) -> Path:
        return self.data_dir / str(self.row["image"])

    @property
    def gold_triples(self) -> list[dict[str, Any]]:
        triples = self.row.get("triples_used") or []
        if isinstance(triples, str):
            return json.loads(triples)
        return list(triples)


def ensure_dataset(cfg: dict[str, Any]) -> Path:
    data_dir = resolve_eval_path(cfg, cfg["dataset"]["data_dir"])
    if _has_jsonl_dataset(data_dir):
        _write_manifest_if_writable(data_dir)
        return data_dir
    if _has_parquet_dataset(data_dir):
        _materialize_parquet_jsonl(data_dir)
        _write_manifest_if_writable(data_dir)
        return data_dir

    repo_id = cfg["dataset"]["repo_id"]
    revision = cfg["dataset"].get("revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub to download ViFoodVQA.") from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
    )
    _copy_dataset_snapshot(snapshot_path, data_dir)
    if _has_parquet_dataset(data_dir) and not _has_jsonl_dataset(data_dir):
        _materialize_parquet_jsonl(data_dir)
    write_manifest(data_dir, repo_id=repo_id, revision=revision)
    return data_dir


def load_split(data_dir: Path, split: str) -> list[VQASample]:
    jsonl_path = data_dir / "data" / f"{split}.jsonl"
    rows = load_jsonl(jsonl_path) if jsonl_path.exists() else load_parquet_split(data_dir, split)
    samples = [VQASample(row=row, split=split, data_dir=data_dir) for row in rows]
    validate_samples(samples)
    return samples


def load_splits(data_dir: Path, splits: Iterable[str]) -> dict[str, list[VQASample]]:
    return {split: load_split(data_dir, split) for split in splits}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def load_parquet_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    paths = _parquet_paths(data_dir, split)
    if not paths:
        raise FileNotFoundError(f"Missing split file: {data_dir / 'data' / f'{split}.jsonl'}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_parquet_rows(path, include_image_bytes=False))
    return rows


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def validate_samples(samples: list[VQASample]) -> None:
    for sample in samples:
        missing = REQUIRED_FIELDS - set(sample.row)
        if missing:
            raise ValueError(f"Sample {sample.row.get('vqa_id')} missing fields: {sorted(missing)}")
        choices = sample.row["choices"]
        if set(choices) != {"A", "B", "C", "D"}:
            raise ValueError(f"Sample {sample.vqa_id} has invalid choices keys")
        if sample.row["answer"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"Sample {sample.vqa_id} has invalid answer")
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing image for sample {sample.vqa_id}: {sample.image_path}")


def write_manifest(data_dir: Path, repo_id: str | None = None, revision: str | None = None) -> Path:
    split_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}
    qtypes: dict[str, dict[str, int]] = {}

    for jsonl_path in sorted((data_dir / "data").glob("*.jsonl")):
        split = jsonl_path.stem
        rows = load_jsonl(jsonl_path)
        split_counts[split] = len(rows)
        image_counts[split] = len({row.get("image_id") for row in rows})
        qtypes[split] = dict(Counter(str(row.get("qtype")) for row in rows))

    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "split_counts": split_counts,
        "image_counts": image_counts,
        "qtypes": qtypes,
    }
    path = data_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_manifest_if_writable(
    data_dir: Path, repo_id: str | None = None, revision: str | None = None
) -> Path | None:
    try:
        return write_manifest(data_dir, repo_id=repo_id, revision=revision)
    except OSError:
        return None


def _has_required_files(data_dir: Path) -> bool:
    return _has_jsonl_dataset(data_dir)


def _has_jsonl_dataset(data_dir: Path) -> bool:
    return (data_dir / "data" / "train.jsonl").exists() and (
        data_dir / "data" / "test.jsonl"
    ).exists() and (data_dir / "images").exists()


def _has_parquet_dataset(data_dir: Path) -> bool:
    return bool(_parquet_paths(data_dir, "train")) and bool(_parquet_paths(data_dir, "test"))


def _parquet_paths(data_dir: Path, split: str) -> list[Path]:
    return sorted((data_dir / "data").glob(f"{split}-*.parquet"))


def _copy_dataset_snapshot(snapshot_path: Path, data_dir: Path) -> None:
    data_src = snapshot_path / "data"
    if not data_src.exists():
        raise FileNotFoundError("Dataset snapshot is missing data/")

    for name in ["data", "images"]:
        src = snapshot_path / name
        if not src.exists():
            continue
        dst = data_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _materialize_parquet_jsonl(data_dir: Path) -> None:
    for split in ["train", "validation", "test"]:
        paths = _parquet_paths(data_dir, split)
        if not paths:
            continue

        jsonl_path = data_dir / "data" / f"{split}.jsonl"
        if jsonl_path.exists():
            continue

        include_image_bytes = not (data_dir / "images").exists()
        with jsonl_path.open("w", encoding="utf-8") as f:
            for path in paths:
                for row in _read_parquet_rows(path, include_image_bytes=include_image_bytes):
                    if include_image_bytes:
                        _write_embedded_image(data_dir, row)
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _read_parquet_rows(path: Path, *, include_image_bytes: bool) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to read the Hugging Face Parquet export.") from exc

    image_column = "image" if include_image_bytes else "image.path"
    columns = [
        "vqa_id",
        "image_id",
        image_column,
        "qtype",
        "question",
        "choices",
        "answer",
        "rationale",
        "triples_used",
    ]
    table = pq.read_table(path, columns=columns)
    return [_normalize_parquet_row(row) for row in table.to_pylist()]


def _normalize_parquet_row(row: dict[str, Any]) -> dict[str, Any]:
    image_value = row.pop("image", None)
    raw_path = row.pop("path", None)
    image_bytes = None

    if isinstance(image_value, dict):
        raw_path = image_value.get("path") or raw_path
        image_bytes = image_value.get("bytes")

    image_path = _normalize_image_path(raw_path, row.get("image_id"))
    row["image"] = image_path
    if image_bytes:
        row["_image_bytes"] = image_bytes
    return row


def _normalize_image_path(raw_path: object, image_id: object) -> str:
    filename = Path(str(raw_path)).name if raw_path else f"{image_id}.jpg"
    return str(Path("images") / filename).replace("\\", "/")


def _write_embedded_image(data_dir: Path, row: dict[str, Any]) -> None:
    image_bytes = row.pop("_image_bytes", None)
    if not image_bytes:
        return

    image_path = data_dir / row["image"]
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image_path.write_bytes(image_bytes)
