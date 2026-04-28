from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

PAGE_SIZE = 1000

# Extra aliases observed in the project code / older generations.
EXTRA_QTYPE_ALIASES: dict[str, str] = {
    "ingredients": "ingredients",
    "ingredient": "ingredients",
    "cooking technique": "cooking_technique",
    "cooking_technique": "cooking_technique",
    "flavor profile": "flavor_profile",
    "flavor_profile": "flavor_profile",
    "taste and flavor profile": "flavor_profile",
    "origin locality": "origin_locality",
    "origin_locality": "origin_locality",
    "origin": "origin_locality",
    "origin_region": "origin_locality",
    "allergen restrictions": "allergen_restrictions",
    "allergen_restrictions": "allergen_restrictions",
    "dietary restrictions": "dietary_restrictions",
    "dietary_restrictions": "dietary_restrictions",
    "ingredient category": "ingredient_category",
    "ingredient_category": "ingredient_category",
    "food pairings": "food_pairings",
    "food_pairings": "food_pairings",
    "side dish": "food_pairings",
    "side_dish": "food_pairings",
    "dish classification": "dish_classification",
    "dish_classification": "dish_classification",
    "dish type": "dish_classification",
    "dish_type": "dish_classification",
    "substitution rules": "substitution_rules",
    "substitution_rules": "substitution_rules",
}

DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "ingredients": "Ingredients",
    "cooking_technique": "Cooking Technique",
    "flavor_profile": "Flavour Profile",
    "origin_locality": "Origin / Region",
    "allergen_restrictions": "Allergen Restrictions",
    "dietary_restrictions": "Dietary Restrictions",
    "ingredient_category": "Ingredient Category",
    "food_pairings": "Food Pairings",
    "dish_classification": "Dish Classification",
    "substitution_rules": "Substitution Rules",
}

GROUP_ORDER = ["1-hop", "2-hop", "Reified"]
SPLIT_ORDER = ["train", "val", "test"]


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def slug_text(value: Any) -> str:
    return norm_text(value).lower().replace("-", "_")


def resolve_question_types_csv(explicit_path: str = "") -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "data" / "question_types.csv",
            cwd / "question_types.csv",
            here / "question_types.csv",
            here / "data" / "question_types.csv",
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    searched = "\n- ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find question_types.csv. Searched:\n- {searched}")


def relation_count_from_path(path_text: str) -> int:
    text = norm_text(path_text)
    if not text:
        return 0

    # Count old-style relations: Dish -[hasIngredient]-> Ingredient
    bracket_rels = text.count("[")
    if bracket_rels > 0:
        return bracket_rels

    # Fallback for arrow-only styles.
    normalized = text.replace("→", "->")
    parts = [p.strip() for p in normalized.split("->") if p.strip()]
    if len(parts) >= 3:
        return (len(parts) - 1) // 2
    return 0


def infer_group(relationship_path: str, canonical_qtype: str) -> str:
    path_lower = norm_text(relationship_path).lower()
    canon = slug_text(canonical_qtype)

    if canon == "substitution_rules" or "hassubrule" in path_lower:
        return "Reified"

    rel_count = relation_count_from_path(relationship_path)
    if rel_count <= 1:
        return "1-hop"
    return "2-hop"


def make_qtype_registry(csv_path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows: list[dict[str, str]] = []
    alias_to_canonical: dict[str, str] = dict(EXTRA_QTYPE_ALIASES)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question_type", "canonical_qtype", "relationship_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"question_types.csv missing required columns: {sorted(missing)}")

        for row in reader:
            canonical = slug_text(row.get("canonical_qtype") or row.get("question_type"))
            if not canonical:
                continue

            supported = slug_text(row.get("supported_in_current_kg"))
            if supported and supported not in {"true", "1", "yes", "y"}:
                continue

            question_type = slug_text(row.get("question_type"))
            relationship_path = norm_text(row.get("relationship_path"))

            alias_to_canonical[canonical] = canonical
            if question_type:
                alias_to_canonical[question_type] = canonical

            rows.append(
                {
                    "question_type": question_type or canonical,
                    "canonical_qtype": canonical,
                    "relationship_path": relationship_path,
                    "group": infer_group(relationship_path, canonical),
                    "display_name": DISPLAY_NAME_OVERRIDES.get(
                        canonical,
                        norm_text(question_type or canonical).replace("_", " ").title(),
                    ),
                }
            )

    # Deduplicate but preserve CSV order by canonical_qtype.
    seen: set[str] = set()
    deduped_rows: list[dict[str, str]] = []
    for row in rows:
        canonical = row["canonical_qtype"]
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped_rows.append(row)

    return deduped_rows, alias_to_canonical


def normalize_split(value: Any) -> str:
    raw = slug_text(value)
    if raw in {"train", "training"}:
        return "train"
    if raw in {"val", "valid", "validate", "validation", "dev"}:
        return "val"
    if raw in {"test", "testing"}:
        return "test"
    return raw


def make_supabase_client():
    from supabase import create_client

    load_dotenv(find_dotenv(usecwd=True) or None)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY missing in .env")
    return create_client(url, key)


def fetch_all_vqa_rows(client, include_dropped: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0

    while True:
        query = client.table("vqa").select("vqa_id,qtype,split,is_drop").order("vqa_id")
        if not include_dropped:
            query = query.eq("is_drop", False)

        resp = query.range(start, start + PAGE_SIZE - 1).execute()
        batch = resp.data or []
        if not batch:
            break

        rows.extend(batch)
        print(f"Loaded {len(rows):,} VQA rows...", flush=True)

        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    return rows


def compute_stats(
    vqa_rows: list[dict[str, Any]],
    qtype_registry: list[dict[str, str]],
    alias_to_canonical: dict[str, str],
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_qtypes: Counter[str] = Counter()
    unknown_splits: Counter[str] = Counter()

    valid_canonicals = {row["canonical_qtype"] for row in qtype_registry}

    for row in vqa_rows:
        raw_qtype = slug_text(row.get("qtype"))
        canonical = alias_to_canonical.get(raw_qtype, raw_qtype)
        split = normalize_split(row.get("split"))

        if canonical not in valid_canonicals:
            unknown_qtypes[raw_qtype or "<empty>"] += 1
            continue

        if split not in SPLIT_ORDER:
            unknown_splits[split or "<empty>"] += 1
            continue

        counts[canonical][split] += 1

    output_rows: list[dict[str, Any]] = []
    for spec in qtype_registry:
        canonical = spec["canonical_qtype"]
        train = counts[canonical]["train"]
        val = counts[canonical]["val"]
        test = counts[canonical]["test"]
        total = train + val + test
        output_rows.append(
            {
                **spec,
                "train": train,
                "val": val,
                "test": test,
                "total": total,
            }
        )

    return output_rows, unknown_qtypes, unknown_splits


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "question_type",
                "canonical_qtype",
                "display_name",
                "relationship_path",
                "train",
                "val",
                "test",
                "total",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def build_latex_table(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    total_train = sum(row["train"] for row in rows)
    total_val = sum(row["val"] for row in rows)
    total_test = sum(row["test"] for row in rows)
    total_all = sum(row["total"] for row in rows)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Dataset split statistics.}",
        r"\label{tab:data-split}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Question Type} & \textbf{Train} & \textbf{Val} & \textbf{Test} & \textbf{Total} \\",
        r"\midrule",
    ]

    first_group_written = False
    for group_name in GROUP_ORDER:
        group_rows = grouped.get(group_name, [])
        if not group_rows:
            continue
        if first_group_written:
            lines.append(r"\midrule")
        first_group_written = True
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{latex_escape(group_name)}}}}} \\")
        for row in group_rows:
            label = latex_escape(row["display_name"])
            lines.append(
                f"{label} & {row['train']} & {row['val']} & {row['test']} & {row['total']} \\\\"
            )

    lines.extend(
        [
            r"\midrule",
            f"\\textbf{{Total}} & {total_train} & {total_val} & {total_test} & {total_all} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_latex_table(rows: list[dict[str, Any]], output_tex: Path) -> None:
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(build_latex_table(rows), encoding="utf-8")


def print_console_summary(rows: list[dict[str, Any]], unknown_qtypes: Counter[str], unknown_splits: Counter[str]) -> None:
    print("\n=== VQA statistics by split and question type ===")
    for group_name in GROUP_ORDER:
        group_rows = [row for row in rows if row["group"] == group_name]
        if not group_rows:
            continue
        print(f"\n[{group_name}]")
        for row in group_rows:
            print(
                f"- {row['display_name']}: "
                f"train={row['train']}, val={row['val']}, test={row['test']}, total={row['total']}"
            )

    total_train = sum(row["train"] for row in rows)
    total_val = sum(row["val"] for row in rows)
    total_test = sum(row["test"] for row in rows)
    total_all = sum(row["total"] for row in rows)
    print(f"\n[Total] train={total_train}, val={total_val}, test={total_test}, total={total_all}")

    if unknown_qtypes:
        print("\n[WARN] qtype not found in question_types.csv (after alias normalization):")
        for key, count in unknown_qtypes.most_common():
            print(f"  - {key}: {count}")

    if unknown_splits:
        print("\n[WARN] split not mapped to train/val/test:")
        for key, count in unknown_splits.most_common():
            print(f"  - {key}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count current VQA rows by split and question type, then export CSV + LaTeX."
    )
    parser.add_argument(
        "--question-types-csv",
        default="",
        help="Path to question_types.csv. If omitted, the script will auto-discover it.",
    )
    parser.add_argument(
        "--include-dropped",
        action="store_true",
        help="Include rows where vqa.is_drop = true. Default: exclude dropped rows.",
    )
    parser.add_argument(
        "--output-csv",
        default="vqa_split_stats.csv",
        help="Where to save the normalized statistics CSV.",
    )
    parser.add_argument(
        "--output-tex",
        default="vqa_split_stats.tex",
        help="Where to save the generated LaTeX table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = resolve_question_types_csv(args.question_types_csv)
    print(f"Using question types from: {csv_path}")

    qtype_registry, alias_to_canonical = make_qtype_registry(csv_path)
    if not qtype_registry:
        raise RuntimeError("No supported qtypes found in question_types.csv")

    client = make_supabase_client()
    vqa_rows = fetch_all_vqa_rows(client, include_dropped=args.include_dropped)
    print(f"Fetched {len(vqa_rows):,} VQA rows for aggregation.")

    rows, unknown_qtypes, unknown_splits = compute_stats(vqa_rows, qtype_registry, alias_to_canonical)

    output_csv = Path(args.output_csv)
    output_tex = Path(args.output_tex)
    write_csv(rows, output_csv)
    write_latex_table(rows, output_tex)
    print_console_summary(rows, unknown_qtypes, unknown_splits)

    print(f"\nSaved CSV : {output_csv.resolve()}")
    print(f"Saved TeX : {output_tex.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
