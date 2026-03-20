from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_base_module():
    base_path = SRC_DIR / "06_generate_vqa.py"
    spec = importlib.util.spec_from_file_location("vqa_base", str(base_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def load_image_ids(path_str: str) -> list[str]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Image ids file not found: {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON image ids file must be a list")
        return [base.norm_text(x) for x in data if base.norm_text(x)]

    ids = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = base.norm_text(line)
        if line:
            ids.append(line)
    return ids


def fetch_rows_by_ids(
    client,
    table: str,
    id_col: str,
    image_col: str,
    desc_col: str,
    items_col: str,
    image_ids: list[str],
    only_approved_images: bool,
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    select_cols = f"{id_col}, {image_col}, {desc_col}, {items_col}, is_checked, is_drop"
    rows: list[dict[str, Any]] = []

    for i in range(0, len(image_ids), batch_size):
        batch = image_ids[i:i + batch_size]
        query = (
            client.table(table)
            .select(select_cols)
            .in_(id_col, batch)
            .not_.is_(items_col, "null")
            .not_.is_(desc_col, "null")
            .order(id_col)
        )
        if only_approved_images:
            query = query.eq("is_checked", True).eq("is_drop", False)

        resp = query.execute()
        rows.extend(resp.data or [])

    # giữ thứ tự theo file input
    order = {img_id: idx for idx, img_id in enumerate(image_ids)}
    rows.sort(key=lambda r: order.get(base.norm_text(r.get(id_col)), 10**9))
    return rows


def append_debug(debug_rows: list[dict[str, str]], image_id: str, qtype: str, fail_stage: str, detail: str):
    debug_rows.append(
        {
            "image_id": base.norm_text(image_id),
            "qtype": base.norm_text(qtype),
            "fail_stage": base.norm_text(fail_stage),
            "detail": base.norm_text(detail)[:1000],
        }
    )


def save_debug_csv(debug_rows: list[dict[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "qtype", "fail_stage", "detail"])
        writer.writeheader()
        writer.writerows(debug_rows)


def describe_invalid_generation(result: dict[str, Any]) -> str:
    if not result:
        return "empty_result"

    choices = result.get("choices") or {}
    choice_keys = sorted(list(choices.keys()))
    answer = base.norm_text(result.get("answer")).upper()
    has_question = bool(base.norm_text(result.get("question_vi")))
    has_reason = bool(base.norm_text(result.get("rationale_vi")))
    nonempty_choices = {k: bool(base.norm_text(v)) for k, v in choices.items()}

    return (
        f"choice_keys={choice_keys}; "
        f"answer={answer or '<empty>'}; "
        f"has_question={has_question}; "
        f"has_reason={has_reason}; "
        f"nonempty_choices={nonempty_choices}"
    )


def generate_one_sample_debug(
    gemini_client,
    kg,
    rng: random.Random,
    image_row: dict[str, Any],
    qmeta: dict[str, str],
    anchor_dish: str,
    top_k: int,
    existing_questions_same_qtype: list[str] | None = None,
    used_answer_keys: set[str] | None = None,
    generation_slot: int = 1,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    logs: list[dict[str, str]] = []
    image_id = image_row["image_id"]
    qtype = qmeta["canonical_qtype"]

    retrieval_query = base.build_retrieval_query(
        qmeta=qmeta,
        dish=anchor_dish,
        image_desc=image_row["image_description"],
        food_items=image_row["food_items"],
    )

    try:
        retrieved = kg.retrieve(items=[anchor_dish], question=retrieval_query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logs.append(
            {
                "image_id": image_id,
                "qtype": qtype,
                "fail_stage": "retrieve_exception",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return None, logs

    if not retrieved:
        logs.append(
            {
                "image_id": image_id,
                "qtype": qtype,
                "fail_stage": "retrieve_empty",
                "detail": f"anchor_dish={anchor_dish}; top_k={top_k}",
            }
        )
        return None, logs

    filtered_rows = base.dedupe_rows(base.filter_rows_by_relationship_path(retrieved, qmeta))
    candidates = base.select_candidates(qmeta, retrieved)

    if not candidates:
        rels = sorted({base.norm_text(r.get("relation")) for r in filtered_rows if base.norm_text(r.get("relation"))})
        logs.append(
            {
                "image_id": image_id,
                "qtype": qtype,
                "fail_stage": "no_candidates",
                "detail": (
                    f"anchor_dish={anchor_dish}; "
                    f"retrieved={len(retrieved)}; "
                    f"filtered_rows={len(filtered_rows)}; "
                    f"relations={rels}"
                ),
            }
        )
        return None, logs

    used_answer_keys = used_answer_keys or set()
    existing_questions_same_qtype = existing_questions_same_qtype or []
    existing_norms = {base.norm_text(q).lower() for q in existing_questions_same_qtype}

    ordered_candidates = [c for c in candidates if c.get("answer_key") not in used_answer_keys]
    if not ordered_candidates:
        ordered_candidates = candidates
        logs.append(
            {
                "image_id": image_id,
                "qtype": qtype,
                "fail_stage": "all_answer_keys_already_used",
                "detail": f"candidate_count={len(candidates)}",
            }
        )

    tried = 0
    for candidate in ordered_candidates[:8]:
        tried += 1
        answer_key = base.norm_text(candidate.get("answer_key"))
        answer_preview = base.norm_text(candidate.get("answer"))[:120]

        prompt_text, retrieved_facts = base.build_indifoodvqa_prompt(
            image_row=image_row,
            qmeta=qmeta,
            candidate=candidate,
            existing_questions_same_qtype=existing_questions_same_qtype,
            generation_slot=generation_slot,
        )

        raw_text = base.call_gemini_generate(gemini_client, prompt_text)
        if not raw_text:
            logs.append(
                {
                    "image_id": image_id,
                    "qtype": qtype,
                    "fail_stage": "gemini_empty",
                    "detail": f"candidate_answer={answer_preview}; answer_key={answer_key}",
                }
            )
            continue

        parsed_items = base.parse_indifoodvqa_output(raw_text)
        if not parsed_items:
            logs.append(
                {
                    "image_id": image_id,
                    "qtype": qtype,
                    "fail_stage": "parse_failed",
                    "detail": f"candidate_answer={answer_preview}; raw_preview={base.norm_text(raw_text)[:250]}",
                }
            )
            continue

        found_valid = False
        triples_used = base.shrink_triples(candidate["triples"])

        for llm_result in parsed_items:
            if not base.validate_generation(llm_result):
                logs.append(
                    {
                        "image_id": image_id,
                        "qtype": qtype,
                        "fail_stage": "validate_failed",
                        "detail": describe_invalid_generation(llm_result),
                    }
                )
                continue

            question_norm = base.norm_text(llm_result.get("question_vi")).lower()
            if question_norm and question_norm in existing_norms:
                logs.append(
                    {
                        "image_id": image_id,
                        "qtype": qtype,
                        "fail_stage": "duplicate_existing_question",
                        "detail": base.norm_text(llm_result.get("question_vi"))[:250],
                    }
                )
                continue

            found_valid = True
            return {
                "image_id": image_row["image_id"],
                "image": image_row["image"],
                "dish": anchor_dish,
                "qtype": qmeta["canonical_qtype"],
                "question_vi": llm_result["question_vi"],
                "choices": llm_result["choices"],
                "answer": llm_result["answer"],
                "rationale_vi": llm_result["rationale_vi"],
                "triples_used": triples_used,
                "retrieved_facts": retrieved_facts,
                "retrieval_query": retrieval_query,
                "food_items": image_row["food_items"],
                "image_description": image_row["image_description"],
                "anchor_entity": candidate.get("anchor"),
                "answer_key": candidate.get("answer_key"),
            }, logs

        if not found_valid:
            logs.append(
                {
                    "image_id": image_id,
                    "qtype": qtype,
                    "fail_stage": "parsed_but_no_valid_item",
                    "detail": f"candidate_answer={answer_preview}; parsed_items={len(parsed_items)}",
                }
            )

    logs.append(
        {
            "image_id": image_id,
            "qtype": qtype,
            "fail_stage": "all_candidates_exhausted",
            "detail": f"candidates_total={len(candidates)}; candidates_tried={tried}",
        }
    )
    return None, logs


def parse_args():
    parser = argparse.ArgumentParser(description="Debug missing VQA image_ids")
    parser.add_argument("--image-ids-file", required=True)
    parser.add_argument("--table", default="image")
    parser.add_argument("--id-col", default="image_id")
    parser.add_argument("--image-col", default="image_url")
    parser.add_argument("--desc-col", default="image_desc")
    parser.add_argument("--items-col", default="food_items")
    parser.add_argument("--question-types-csv", default=str(base.QUESTION_TYPES_FILE))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "vqa_debug"))
    parser.add_argument("--qtypes-per-image", type=int, default=0, help="0 = use all selected qtypes")
    parser.add_argument("--qtypes", nargs="*", default=[])
    parser.add_argument("--top-k", type=int, default=base.TOP_K)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--questions-per-qtype", type=int, default=1)
    parser.add_argument("--only-approved-images", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_csv = output_dir / "debug_failures.csv"
    generated_json = output_dir / "debug_generated_vqa.json"

    image_ids = load_image_ids(args.image_ids_file)
    if not image_ids:
        raise RuntimeError("No image ids found in file")

    all_qtypes = base.load_question_types(Path(args.question_types_csv))
    qmeta_by_key = {q["canonical_qtype"]: q for q in all_qtypes}

    if args.qtypes:
        requested_qtypes = []
        for raw in args.qtypes:
            canon = base.QTYPE_ALIASES.get(base.norm_text(raw).lower(), base.norm_text(raw).lower())
            if canon not in qmeta_by_key:
                print(f"[WARN] Skip unsupported qtype: {raw} -> {canon}")
                continue
            requested_qtypes.append(qmeta_by_key[canon])
    else:
        requested_qtypes = list(qmeta_by_key.values())

    if not requested_qtypes:
        raise RuntimeError("No supported question types selected")

    supabase = base.make_supabase_client()
    gemini_client = base.make_gemini_client()
    kg = base.make_retriever(device=args.device)

    generated: list[dict[str, Any]] = []
    debug_rows: list[dict[str, str]] = []
    questions_by_pair: dict[tuple[str, str], list[str]] = {}
    answers_by_pair: dict[tuple[str, str], set[str]] = {}
    question_keys: set[str] = set()

    try:
        embedded_edges = base.count_embedded_edges(kg)
        if embedded_edges == 0:
            raise RuntimeError("No relationship embeddings found in Neo4j")

        substitution_emb = base.count_substitution_embeddings(kg)
        if substitution_emb == 0 and any(q["canonical_qtype"] == "substitution_rules" for q in requested_qtypes):
            print(
                "[WARN] substitution_rules is disabled because fromIngredient/toIngredient embeddings are missing."
            )
            requested_qtypes = [q for q in requested_qtypes if q["canonical_qtype"] != "substitution_rules"]

        dish_aliases = base.fetch_all_dishes(kg)

        print(f"Loaded {len(dish_aliases)} dishes from Neo4j")
        print(f"Embedded relationships: {embedded_edges}")
        print("Question types:", ", ".join(q["canonical_qtype"] for q in requested_qtypes))
        print(f"Target image ids: {len(image_ids)}")
        print(f"Questions per qtype: {args.questions_per_qtype}")

        rows = fetch_rows_by_ids(
            client=supabase,
            table=args.table,
            id_col=args.id_col,
            image_col=args.image_col,
            desc_col=args.desc_col,
            items_col=args.items_col,
            image_ids=image_ids,
            only_approved_images=args.only_approved_images,
        )

        print(f"Fetched image rows from Supabase: {len(rows)}")

        fetched_ids = {base.norm_text(r.get(args.id_col)) for r in rows}
        not_found_ids = [img_id for img_id in image_ids if img_id not in fetched_ids]
        for img_id in not_found_ids:
            append_debug(debug_rows, img_id, "", "not_fetched_from_supabase", "No matching approved row returned")

        attempted_images = 0

        for raw in rows:
            image_row = {
                "image_id": base.norm_text(raw.get(args.id_col)),
                "image": base.norm_text(raw.get(args.image_col)),
                "image_description": base.norm_text(raw.get(args.desc_col)),
                "food_items": [base.norm_text(x) for x in (raw.get(args.items_col) or []) if base.norm_text(x)],
            }
            attempted_images += 1

            if not image_row["image_id"] or not image_row["image_description"] or not image_row["food_items"]:
                append_debug(
                    debug_rows,
                    image_row["image_id"],
                    "",
                    "missing_required_fields",
                    f"has_image_id={bool(image_row['image_id'])}; "
                    f"has_desc={bool(image_row['image_description'])}; "
                    f"food_items={len(image_row['food_items'])}",
                )
                continue

            anchor_dish = base.choose_anchor_dish(image_row["food_items"], dish_aliases)
            if not anchor_dish:
                append_debug(
                    debug_rows,
                    image_row["image_id"],
                    "",
                    "no_anchor_dish",
                    f"food_items={image_row['food_items'][:12]}",
                )
                continue

            qtypes_for_image = requested_qtypes[:]
            rng.shuffle(qtypes_for_image)
            if args.qtypes_per_image and args.qtypes_per_image > 0:
                qtypes_for_image = qtypes_for_image[: args.qtypes_per_image]

            for qmeta in qtypes_for_image:
                pair_key = (image_row["image_id"], qmeta["canonical_qtype"])
                existing_questions = questions_by_pair.get(pair_key, [])
                used_answer_keys = answers_by_pair.get(pair_key, set())

                while len(existing_questions) < args.questions_per_qtype:
                    sample, logs = generate_one_sample_debug(
                        gemini_client=gemini_client,
                        kg=kg,
                        rng=rng,
                        image_row=image_row,
                        qmeta=qmeta,
                        anchor_dish=anchor_dish,
                        top_k=args.top_k,
                        existing_questions_same_qtype=existing_questions,
                        used_answer_keys=used_answer_keys,
                        generation_slot=len(existing_questions) + 1,
                    )

                    debug_rows.extend(logs)

                    if not sample:
                        break

                    qsig = base.question_signature(sample["image_id"], sample["qtype"], sample["question_vi"])
                    if qsig in question_keys:
                        append_debug(
                            debug_rows,
                            sample["image_id"],
                            sample["qtype"],
                            "duplicate_question_signature",
                            base.norm_text(sample["question_vi"])[:250],
                        )
                        break

                    generated.append(sample)
                    question_keys.add(qsig)
                    existing_questions = questions_by_pair.setdefault(pair_key, [])
                    existing_questions.append(sample["question_vi"])
                    if base.norm_text(sample.get("answer_key")):
                        answers_by_pair.setdefault(pair_key, set()).add(base.norm_text(sample.get("answer_key")))
                        used_answer_keys = answers_by_pair[pair_key]

                    print(
                        f"  ✓ {sample['image_id']}::{sample['qtype']}::{len(existing_questions)} "
                        f"-> {sample['answer']} | {sample['question_vi'][:90]}"
                    )

        generated_json.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
        save_debug_csv(debug_rows, debug_csv)

        stage_counts = Counter(row["fail_stage"] for row in debug_rows)
        print("\n-- Done --")
        print(f"Attempted images: {attempted_images}")
        print(f"Generated samples: {len(generated)}")
        print(f"Debug rows: {len(debug_rows)}")
        print(f"Saved generated JSON: {generated_json}")
        print(f"Saved debug CSV: {debug_csv}")

        print("\nTop fail stages:")
        for stage, count in stage_counts.most_common(20):
            print(f"  {stage}: {count}")

    finally:
        kg.close()


if __name__ == "__main__":
    main()
