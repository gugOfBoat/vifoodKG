from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    load_config,
    resolve_eval_path,
    selected_conditions,
    selected_models,
)
from .data import VQASample, ensure_dataset, load_split, write_jsonl_row
from .metrics import retrieval_scores
from .models import make_model
from .parsing import parse_answer_letter, parse_classifier_response
from .prompts import CANONICAL_QTYPES, build_answer_messages, build_classifier_messages
from .retrieval import EvaluationRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViFoodVQA evaluation.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-ids", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_eval_path(cfg, cfg["paths"]["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    data_dir = ensure_dataset(cfg)
    samples = load_split(data_dir, cfg["dataset"]["test_split"])
    train_samples = load_split(data_dir, "train")
    if args.sample_ids:
        samples = _select_samples_by_id(samples, args.sample_ids)
    if args.limit:
        samples = samples[: args.limit]

    shots = _select_shots(train_samples, cfg["evaluation"]["fixed_shot_vqa_ids"])
    models = selected_models(cfg, args.models)
    conditions = selected_conditions(cfg, args.conditions)
    kg_needed = any(condition.get("knowledge") == "retrieved" for condition in conditions)

    retriever = EvaluationRetriever(cfg, device=args.device) if kg_needed else None
    try:
        for model_name, model_cfg in models.items():
            model = make_model(model_name, model_cfg)
            classifier_cache = _load_cache(run_dir / "classifier" / f"{model_name}.jsonl")
            for condition in conditions:
                _run_condition(
                    cfg=cfg,
                    run_dir=run_dir,
                    model_name=model_name,
                    model=model,
                    condition=condition,
                    samples=samples,
                    shots=shots,
                    classifier_cache=classifier_cache,
                    retriever=retriever,
                    resume=args.resume,
                )
    finally:
        if retriever:
            retriever.close()


def _run_condition(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    model_name: str,
    model: Any,
    condition: dict[str, Any],
    samples: list[VQASample],
    shots: list[VQASample],
    classifier_cache: dict[int, dict[str, Any]],
    retriever: EvaluationRetriever | None,
    resume: bool,
) -> None:
    out_path = run_dir / "predictions" / f"{model_name}__{condition['name']}.jsonl"
    done = _completed_ids(out_path) if resume else set()
    top_k = int(cfg["evaluation"]["top_k"])

    for sample in samples:
        if sample.vqa_id in done:
            continue

        classifier = {"qtype": None, "food_items": []}
        if condition.get("knowledge") == "retrieved":
            classifier = _classify_sample(
                cfg,
                run_dir,
                model_name,
                model,
                sample,
                classifier_cache,
            )

        retrieved = _knowledge_for_condition(sample, condition, classifier, retriever)
        retrieval_metric = (
            retrieval_scores(retrieved, sample.gold_triples, k=top_k)
            if condition.get("knowledge") in {"retrieved", "oracle"}
            else {}
        )
        prompt_shots = shots[: int(condition.get("shots") or 0)]
        messages = build_answer_messages(sample, prompt_shots, retrieved, top_k=top_k)

        started = time.perf_counter()
        raw_response = model.generate(
            messages,
            max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]),
            temperature=float(cfg["evaluation"]["temperature"]),
        )
        latency = time.perf_counter() - started
        answer, parse_status = parse_answer_letter(raw_response)

        row = {
            "model": model_name,
            "condition": condition["name"],
            "retrieval_strategy": condition.get("retrieval_strategy"),
            "shots": int(condition.get("shots") or 0),
            "vqa_id": sample.vqa_id,
            "image_id": sample.row["image_id"],
            "question": sample.row["question"],
            "choices": sample.row["choices"],
            "qtype_gold": sample.row["qtype"],
            "qtype_pred": classifier.get("qtype"),
            "food_items_pred": classifier.get("food_items", []),
            "answer_gold": sample.row["answer"],
            "answer_pred": answer,
            "correct": answer == sample.row["answer"],
            "parse_status": parse_status,
            "retrieved_triples": retrieved,
            "raw_response": raw_response,
            "latency_s": round(latency, 4),
            **retrieval_metric,
        }
        write_jsonl_row(out_path, row)


def _classify_sample(
    cfg: dict[str, Any],
    run_dir: Path,
    model_name: str,
    model: Any,
    sample: VQASample,
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if sample.vqa_id in cache:
        return cache[sample.vqa_id]

    messages = build_classifier_messages(sample)
    raw = model.generate(
        messages,
        max_new_tokens=128,
        temperature=float(cfg["evaluation"]["temperature"]),
        response_format={"type": "json_object"},
    )
    parsed, status = parse_classifier_response(raw, set(CANONICAL_QTYPES))
    row = {
        "vqa_id": sample.vqa_id,
        "qtype_gold": sample.row["qtype"],
        "qtype_pred": parsed.get("qtype"),
        "food_items": parsed.get("food_items", []),
        "status": status,
        "raw_response": raw,
    }
    write_jsonl_row(run_dir / "classifier" / f"{model_name}.jsonl", row)
    cache[sample.vqa_id] = {"qtype": row["qtype_pred"], "food_items": row["food_items"]}
    return cache[sample.vqa_id]


def _knowledge_for_condition(
    sample: VQASample,
    condition: dict[str, Any],
    classifier: dict[str, Any],
    retriever: EvaluationRetriever | None,
) -> list[dict[str, Any]]:
    knowledge = condition.get("knowledge")
    if knowledge == "none":
        return []
    if knowledge == "oracle":
        return sample.gold_triples
    if knowledge == "retrieved":
        if retriever is None:
            raise RuntimeError("Retriever is required for retrieved KG conditions.")
        return retriever.retrieve(sample, condition["retrieval_strategy"], classifier)
    raise ValueError(f"Unknown knowledge condition: {knowledge}")


def _select_shots(train_samples: list[VQASample], shot_ids: list[int]) -> list[VQASample]:
    by_id = {sample.vqa_id: sample for sample in train_samples}
    missing = [vqa_id for vqa_id in shot_ids if vqa_id not in by_id]
    if missing:
        raise KeyError(f"Configured shot sample(s) not found in train split: {missing}")
    return [by_id[vqa_id] for vqa_id in shot_ids]


def _select_samples_by_id(samples: list[VQASample], sample_ids: list[int]) -> list[VQASample]:
    by_id = {sample.vqa_id: sample for sample in samples}
    missing = [vqa_id for vqa_id in sample_ids if vqa_id not in by_id]
    if missing:
        raise KeyError(f"Requested sample(s) not found in evaluation split: {missing}")
    return [by_id[vqa_id] for vqa_id in sample_ids]


def _load_cache(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[int(row["vqa_id"])] = {
                "qtype": row.get("qtype_pred"),
                "food_items": row.get("food_items", []),
            }
    return cache


def _completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(row["vqa_id"]) for row in _read_jsonl(path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
