from __future__ import annotations

"""
ViFoodKG — Step 6: Generate ViFoodVQA samples
==============================================

Pipeline
--------
1. Read approved images from Supabase (`image.is_checked = true` and `image.is_drop = false`)
2. Load question types from `data/question_types.csv`
3. Use `query.py::KGRetriever` to retrieve top-K triples from Neo4j
4. Filter triples by qtype and build one grounded candidate answer
5. Give image description + retrieved triples + fixed 4 choices to Gemini
6. Save checkpoint + final JSON

Important
---------
- This script depends on edge embeddings stored in Neo4j by `05_kg_vectorizer.py`.
- `query.py` only returns relationships whose `embedding` property is not null.
- With the CURRENT `05_kg_vectorizer.py`, `substitution_rules` will usually be skipped,
  because `fromIngredient` / `toIngredient` edges are not vectorized yet.

Usage
-----
  python src/06_generate_vqa.py
  python src/06_generate_vqa.py --limit-images 20 --qtypes-per-image 2
  python src/06_generate_vqa.py --qtypes ingredients origin side_dish
  python src/06_generate_vqa.py --start-page 3 --top-k 10
  python src/06_generate_vqa.py --output-dir /content/drive/MyDrive/ViFoodKG_outputs/vqa
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

QUESTION_TYPES_FILE = PROJECT_ROOT / "data" / "question_types.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "vqa"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "generated_vqa.json"
DEFAULT_PROGRESS_FILE = DEFAULT_OUTPUT_DIR / "_generate_vqa_progress.json"

GEMINI_MODEL = os.getenv("VQA_GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
PAGE_SIZE = 500
TOP_K = 8
DEFAULT_SEED = 42

# qtypes that can be supported by the CURRENT retriever + vectorizer stack
SUPPORTED_QTYPES = {
    "ingredients",
    "cooking_technique",
    "flavor_profile",
    "origin_locality",
    "allergen_restrictions",
    "dietary_restrictions",
    "ingredient_category",
    "food_pairings",
    "dish_classification",
    "substitution_rules",
}

QTYPE_ALIASES = {
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
    "substitution rules": "substitution_rules",
    "substitution_rules": "substitution_rules",
}

TARGET_LABEL_TO_NODE = {
    "Ingredient": "Ingredient",
    "SideDish": "SideDish",
    "Condiment": "Condiment",
    "Region": "Region",
    "DishType": "DishType",
    "IngredientCategory": "IngredientCategory",
    "Allergen": "Allergen",
    "CookingTechnique": "CookingTechnique",
    "FlavorProfile": "FlavorProfile",
    "DietaryTag": "DietaryTag",
}

DIETARY_STATEMENTS = {
    "plant_based": "Phù hợp với chế độ ăn thuần thực vật.",
    "animal_product": "Không phù hợp với chế độ ăn thuần thực vật.",
    "chay": "Phù hợp với chế độ ăn thuần thực vật.",
    "mặn": "Không phù hợp với chế độ ăn thuần thực vật.",
}

INDIFOODVQA_PROMPT_TEMPLATE = """
You are a Vietnamese food specialist AI visual assistant, and you are seeing a single image. What you see are provided with some sentences, describing the same image you are looking at. Answer all questions as you are seeing the image.
Description : {image_description}
Use the following facts when generating the questions, given in the form of triples:
{kg_triples}
Give an output with 4 parts, with each part separated by 2 blank lines: a question (name it Question, and give the question in the next line), 4 possible answer choices (name it Answer Choices, with choices A, B, C and D in separate lines), the correct answer to that question (name it Correct Answer, out of A, B, C and D), and a reason for that answer (name it Reason, limited to 1 paragraph). Ask diverse questions and give corresponding answers. Give me a question as output. Only include questions that have definite answers :
(1) one can see the content in the image that the question asks about and can answer confidently
(2) one can determine confidently from the image that it is not in the image. Do not ask any question that cannot be answered confidently.
The question should be about {question_type} of the food items in the image This includes details about {detailed_information}. The question should involve complex ideas like relative positions of the objects, the shapes and colors of the objects, and so on. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Nowhere should it be mentioned that a description or some external knowledge has been provided. Act like you can see the image, and create complex questions requiring multiple steps of reasoning.
The knowledge triples do not describe the image. If any of the given knowledge triples are used to generate the question, then do not mention the entities given in the knowledge triple in the Question or Answer Choices. Ensure that in the case that any knowledge triple is used, the question is not answerable without using this external knowledge. The knowledge used to generate the question can only be mentioned in the Reason field.
Also, create questions about both the main dish and the side dish. Try to include the relative position between the items as a part of the question. But keep the main question about {question_type} of the food items. Do not bold anything (keep everything in normal font), and do not number the questions. The question and each answer choice should be in a new line. Make sure the questions involve reasoning to answer. The output should contain 5 such diverse questions (5 questions with given format). Do not mention the word "knowledge" or "triples" or "description" anywhere. Don't include any numbers anywhere.
Write all output in Vietnamese.
""".strip()


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def slug(text: str) -> str:
    cleaned = norm_text(text).lower().replace("đ", "d")
    cleaned = unicodedata.normalize("NFD", cleaned)
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return cleaned


def truthy(value: Any) -> bool:
    return norm_text(value).lower() in {"1", "true", "yes", "y"}


def relation_to_vi(relation: str) -> str:
    mapping = {
        "hasIngredient": "có nguyên liệu",
        "servedWith": "thường ăn kèm với",
        "originRegion": "có nguồn gốc từ",
        "dishType": "thuộc loại món",
        "ingredientCategory": "thuộc nhóm nguyên liệu",
        "hasAllergen": "có chất gây dị ứng",
        "cookingTechnique": "được chế biến bằng",
        "flavorProfile": "có hương vị",
        "hasDietaryTag": "mang nhãn chế độ ăn",
        "hasSubRule": "có quy tắc thay thế",
        "fromIngredient": "thay nguyên liệu gốc",
        "toIngredient": "bằng nguyên liệu",
    }
    return mapping.get(relation, relation)



def parse_relationship_path(path_text: str) -> list[str]:
    text = norm_text(path_text)
    if not text:
        return []

    relations: list[str] = []
    seen: set[str] = set()

    # Old style: Dish -[hasIngredient]-> Ingredient
    for rel in re.findall(r"\[\s*([^\]]+?)\s*\]", text):
        rel = norm_text(rel)
        if rel and rel not in seen:
            seen.add(rel)
            relations.append(rel)

    # New style: Dish → hasSubRule → SubRule_X / Dish -> Ingredient -[hasAllergen]-> Allergen
    normalized = (
        text.replace("→", " -> ")
        .replace("=>", " -> ")
        .replace(" - ", " ")
    )
    tokens = [norm_text(tok) for tok in re.split(r"\s*->\s*", normalized) if norm_text(tok)]
    for idx in range(1, len(tokens), 2):
        rel = tokens[idx]
        if rel and rel not in seen and re.match(r"^[A-Za-z][A-Za-z0-9_]*$", rel):
            seen.add(rel)
            relations.append(rel)

    return relations

def verbalize_triples(triples: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for t in triples:
        subject = norm_text(t.get("subject"))
        relation = norm_text(t.get("relation"))
        target = norm_text(t.get("target"))
        via = norm_text(t.get("via"))
        if via:
            lines.append(f"{subject} ; {relation} ; {target} (qua {via})")
        else:
            lines.append(f"{subject} ; {relation} ; {target}")
    return lines


def shrink_triples(triples: list[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for t in triples:
        subject = norm_text(t.get("subject"))
        relation = norm_text(t.get("relation"))
        target = norm_text(t.get("target"))
        if not subject or not relation or not target:
            continue
        key = (subject, relation, target)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "subject": subject,
            "relation": relation,
            "target": target,
        })
    return result


def load_question_types(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Question types CSV not found: {path}")

    qtypes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical_raw = norm_text(
                row.get("canonical_qtype") or row.get("qtype") or row.get("question_type") or ""
            )
            if not canonical_raw:
                continue

            canonical = QTYPE_ALIASES.get(canonical_raw.lower(), canonical_raw.lower())
            if canonical not in SUPPORTED_QTYPES:
                continue

            supported_flag = row.get("supported_in_current_kg")
            if supported_flag is not None and supported_flag != "" and not truthy(supported_flag):
                continue

            relationship_path = norm_text(row.get("relationship_path") or row.get("relationship"))
            relationship_sequence = parse_relationship_path(relationship_path)
            if not relationship_sequence:
                print(f"[WARN] Skip qtype without parseable relationship path: {canonical}")
                continue

            qtypes.append(
                {
                    "question_type": norm_text(row.get("question_type") or canonical),
                    "canonical_qtype": canonical,
                    "relationship_path": relationship_path,
                    "relationship_sequence": relationship_sequence,
                    "primary_relation": relationship_sequence[-1],
                    "keywords": norm_text(row.get("keywords")),
                    "description": norm_text(row.get("description")),
                    "detail_description": norm_text(
                        row.get("detail_description") or row.get("detailed_description")
                    ),
                }
            )
    return qtypes


def resolve_output_paths(output_dir: str | Path | None = None) -> tuple[Path, Path, Path]:
    base_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    return (
        base_dir,
        base_dir / "generated_vqa.json",
        base_dir / "_generate_vqa_progress.json",
    )


def load_progress(progress_file: Path) -> dict[str, Any]:
    if progress_file.exists():
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        data.setdefault("page", 0)
        data.setdefault("generated", [])
        data.setdefault("question_keys", [])
        return data
    return {"page": 0, "generated": [], "question_keys": []}


def save_progress(progress: dict[str, Any], output_dir: Path, progress_file: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def make_supabase_client():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY missing in .env")
    return create_client(url, key)


def make_gemini_client():
    from google import genai

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in .env")
    return genai.Client(api_key=key)


def fetch_image_rows(
    client,
    table: str,
    id_col: str,
    image_col: str,
    desc_col: str,
    items_col: str,
    page: int,
    size: int,
    only_approved_images: bool,
    start_image_id: str = "",
    end_image_id: str = "",
    allowed_image_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    select_cols = f"{id_col}, {image_col}, {desc_col}, {items_col}, is_checked, is_drop"
    query = (
        client.table(table)
        .select(select_cols)
        .not_.is_(items_col, "null")
        .not_.is_(desc_col, "null")
        .order(id_col)
    )
    if only_approved_images:
        query = query.eq("is_checked", True).eq("is_drop", False)
    if start_image_id:
        query = query.gte(id_col, start_image_id)
    if end_image_id:
        query = query.lte(id_col, end_image_id)
    if allowed_image_ids:
        query = query.in_(id_col, sorted(allowed_image_ids))
    resp = query.range(page * size, (page + 1) * size - 1).execute()
    return resp.data or []


def load_allowed_image_ids(path_str: str) -> set[str]:
    path_str = norm_text(path_str)
    if not path_str:
        return set()

    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Image id file not found: {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("image_ids json must be a list")
        return {norm_text(x) for x in data if norm_text(x)}

    return {
        norm_text(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if norm_text(line)
    }


# ---------------------------------------------------------------------------
# Neo4j / query.py helpers
# ---------------------------------------------------------------------------

def make_retriever(device: str = "auto"):
    # file lives in the same src/ folder as query.py in the real project
    from query import KGRetriever

    return KGRetriever(device=device)


def fetch_all_dishes(kg) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with kg._driver.session() as session:
        rows = session.run("MATCH (d:Dish) RETURN d.name AS name")
        for row in rows:
            name = norm_text(row["name"])
            if name:
                mapping[slug(name)] = name
    return mapping


def count_embedded_edges(kg) -> int:
    with kg._driver.session() as session:
        row = session.run(
            "MATCH ()-[r]-() WHERE r.embedding IS NOT NULL RETURN count(r) AS c"
        ).single()
        return int(row["c"]) if row and row["c"] is not None else 0


def count_substitution_embeddings(kg) -> int:
    with kg._driver.session() as session:
        row = session.run(
            "MATCH ()-[r:fromIngredient|toIngredient]-() WHERE r.embedding IS NOT NULL RETURN count(r) AS c"
        ).single()
        return int(row["c"]) if row and row["c"] is not None else 0


def choose_anchor_dish(food_items: list[str], dish_aliases: dict[str, str]) -> str | None:
    for item in food_items:
        key = slug(item)
        if key in dish_aliases:
            return dish_aliases[key]
    return None


def build_retrieval_query(qmeta: dict[str, Any], dish: str, image_desc: str, food_items: list[str]) -> str:
    keywords = qmeta.get("keywords", "")
    detail = qmeta.get("detail_description", "")
    rel_path = qmeta.get("relationship_path", "")
    rel_seq = ", ".join(qmeta.get("relationship_sequence", []))
    visible = ", ".join(food_items[:12])
    short_desc = image_desc[:240]
    return (
        f"Loại câu hỏi: {qmeta['canonical_qtype']}. "
        f"Món neo: {dish}. "
        f"Từ khóa: {keywords}. "
        f"Mô tả: {detail}. "
        f"Đường đi quan hệ: {rel_path}. "
        f"Chuỗi quan hệ: {rel_seq}. "
        f"Vật thể thay được: {visible}. "
        f"Ngữ cảnh ảnh: {short_desc}"
    )


# ---------------------------------------------------------------------------
# Candidate construction from query.py retrieval results
# ---------------------------------------------------------------------------


def filter_rows_by_relationship_path(rows: list[dict[str, Any]], qmeta: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_relations = set(qmeta.get("relationship_sequence") or [])
    if not allowed_relations:
        return rows
    return [row for row in rows if norm_text(row.get("relation")) in allowed_relations]

def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            norm_text(row.get("subject")),
            norm_text(row.get("relation")),
            norm_text(row.get("target")),
            norm_text(row.get("via")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def simple_candidates(rows: list[dict[str, Any]], relation: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("relation") != relation:
            continue
        answer = norm_text(row.get("target"))
        target_type = norm_text(row.get("target_type"))
        if not answer:
            continue
        candidates.append(
            {
                "answer": answer,
                "answer_key": answer,
                "answer_label": target_type,
                "anchor": norm_text(row.get("via")) or answer,
                "triples": [
                    {
                        "subject": norm_text(row.get("subject")),
                        "relation": relation,
                        "target": answer,
                    }
                ],
                "retrieved_rows": [row],
                "score": float(row.get("score", 0.0)),
            }
        )
    candidates.sort(key=lambda x: (-x["score"], x["answer"]))
    return candidates


def two_hop_candidates(rows: list[dict[str, Any]], relation: str, answer_label: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("relation") != relation:
            continue
        subject = norm_text(row.get("subject"))
        via = norm_text(row.get("via"))
        target = norm_text(row.get("target"))
        if not subject or not via or not target:
            continue
        triples = [
            {"subject": subject, "relation": "hasIngredient", "target": via},
            {"subject": via, "relation": relation, "target": target},
        ]
        candidates.append(
            {
                "answer": target,
                "answer_key": f"{via}|{target}",
                "answer_label": answer_label,
                "anchor": via,
                "triples": triples,
                "retrieved_rows": [row],
                "score": float(row.get("score", 0.0)),
            }
        )
    candidates.sort(key=lambda x: (-x["score"], x["anchor"], x["answer"]))
    return candidates


def dietary_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("relation") != "hasDietaryTag":
            continue
        subject = norm_text(row.get("subject"))
        via = norm_text(row.get("via"))
        target = norm_text(row.get("target")).lower()
        if target not in DIETARY_STATEMENTS:
            continue
        answer_text = DIETARY_STATEMENTS[target]
        triples = [
            {"subject": subject, "relation": "hasIngredient", "target": via},
            {"subject": via, "relation": "hasDietaryTag", "target": target},
        ]
        candidates.append(
            {
                "answer": answer_text,
                "answer_key": target,
                "answer_label": "DietaryStatement",
                "anchor": via,
                "triples": triples,
                "retrieved_rows": [row],
                "score": float(row.get("score", 0.0)),
            }
        )
    candidates.sort(key=lambda x: (-x["score"], x["anchor"]))
    return candidates


def substitution_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        relation = norm_text(row.get("relation"))
        via = norm_text(row.get("via"))
        if relation not in {"fromIngredient", "toIngredient"} or not via:
            continue
        grouped.setdefault(
            via,
            {
                "subject": norm_text(row.get("subject")),
                "via": via,
                "from": None,
                "to": None,
                "rows": [],
                "score": 0.0,
            },
        )
        entry = grouped[via]
        entry["rows"].append(row)
        entry["score"] = max(entry["score"], float(row.get("score", 0.0)))
        if relation == "fromIngredient":
            entry["from"] = norm_text(row.get("target"))
        if relation == "toIngredient":
            entry["to"] = norm_text(row.get("target"))

    candidates: list[dict[str, Any]] = []
    for entry in grouped.values():
        from_ing = entry["from"]
        to_ing = entry["to"]
        subject = entry["subject"]
        via = entry["via"]
        if not from_ing or not to_ing or not subject:
            continue
        answer_text = f"Có thể thay {from_ing} bằng {to_ing}."
        candidates.append(
            {
                "answer": answer_text,
                "answer_key": f"{from_ing}|{to_ing}",
                "answer_label": "SubstitutionPair",
                "anchor": from_ing,
                "triples": [
                    {"subject": subject, "relation": "hasSubRule", "target": via},
                    {"subject": via, "relation": "fromIngredient", "target": from_ing},
                    {"subject": via, "relation": "toIngredient", "target": to_ing},
                ],
                "retrieved_rows": entry["rows"],
                "score": entry["score"],
            }
        )
    candidates.sort(key=lambda x: (-x["score"], x["answer_key"]))
    return candidates


def select_candidates(qmeta: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qtype = qmeta["canonical_qtype"]
    rows = dedupe_rows(filter_rows_by_relationship_path(rows, qmeta))

    if qtype == "ingredients":
        return simple_candidates(rows, "hasIngredient")
    if qtype == "food_pairings":
        return simple_candidates(rows, "servedWith")
    if qtype == "origin_locality":
        return simple_candidates(rows, "originRegion")
    if qtype == "dish_classification":
        return simple_candidates(rows, "dishType")
    if qtype == "cooking_technique":
        return simple_candidates(rows, "cookingTechnique")
    if qtype == "flavor_profile":
        return simple_candidates(rows, "flavorProfile")
    if qtype == "ingredient_category":
        return two_hop_candidates(rows, "ingredientCategory", "IngredientCategory")
    if qtype == "allergen_restrictions":
        return two_hop_candidates(rows, "hasAllergen", "Allergen")
    if qtype == "dietary_restrictions":
        return dietary_candidates(rows)
    if qtype == "substitution_rules":
        return substitution_candidates(rows)
    return []


# ---------------------------------------------------------------------------
# Distractor helpers
# ---------------------------------------------------------------------------

def get_random_node_names(kg, label: str, exclude: list[str], limit: int) -> list[str]:
    safe_label = re.sub(r"[^A-Za-z0-9_]", "", label)
    if not safe_label:
        return []
    query = (
        f"MATCH (n:{safe_label}) WHERE NOT n.name IN $exclude "
        "RETURN n.name AS name ORDER BY rand() LIMIT $limit"
    )
    with kg._driver.session() as session:
        rows = session.run(query, exclude=exclude, limit=limit)
        return [norm_text(row["name"]) for row in rows if norm_text(row["name"])]


def get_random_substitution_pairs(kg, exclude: list[str], limit: int) -> list[str]:
    query = """
    MATCH (d:Dish)-[:hasSubRule]->(sr:SubstitutionRule)-[:fromIngredient]->(src:Ingredient)
    MATCH (sr)-[:toIngredient]->(dst:Ingredient)
    RETURN DISTINCT src.name AS src, dst.name AS dst
    ORDER BY rand()
    LIMIT $limit
    """
    result: list[str] = []
    with kg._driver.session() as session:
        rows = session.run(query, limit=max(limit * 4, 12))
        for row in rows:
            text = f"Có thể thay {norm_text(row['src'])} bằng {norm_text(row['dst'])}."
            if text in exclude or text in result:
                continue
            result.append(text)
            if len(result) >= limit:
                break
    return result


def build_choices(candidate: dict[str, Any], kg, rng: random.Random) -> tuple[dict[str, str], str] | None:
    correct = norm_text(candidate["answer"])
    label = norm_text(candidate["answer_label"])

    if label == "DietaryStatement":
        opposite = (
            "Không phù hợp với chế độ ăn thuần thực vật."
            if correct == "Phù hợp với chế độ ăn thuần thực vật."
            else "Phù hợp với chế độ ăn thuần thực vật."
        )
        distractors = [
            opposite,
            "Chỉ phù hợp với người tránh gluten.",
            "Chỉ phù hợp khi bỏ toàn bộ món ăn kèm.",
        ]
    elif label == "SubstitutionPair":
        distractors = get_random_substitution_pairs(kg, exclude=[correct], limit=3)
    else:
        node_label = TARGET_LABEL_TO_NODE.get(label, label)
        distractors = get_random_node_names(kg, node_label, exclude=[correct], limit=8)
        distractors = [d for d in distractors if d and d != correct][:3]

    unique_options = [correct]
    for value in distractors:
        value = norm_text(value)
        if value and value not in unique_options:
            unique_options.append(value)
    if len(unique_options) < 4:
        return None
    unique_options = unique_options[:4]

    rng.shuffle(unique_options)
    letters = ["A", "B", "C", "D"]
    choices = {letter: text for letter, text in zip(letters, unique_options, strict=True)}
    answer_letter = next(letter for letter, text in choices.items() if text == correct)
    return choices, answer_letter


# ---------------------------------------------------------------------------
# Gemini generation
# ---------------------------------------------------------------------------

def format_kg_triples_for_prompt(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        subject = norm_text(row.get("subject"))
        relation = norm_text(row.get("relation"))
        target = norm_text(row.get("target"))
        if subject and relation and target:
            lines.append(f"({subject}, {relation}, {target})")
    return "\n".join(lines)


def build_indifoodvqa_prompt(
    image_row: dict[str, Any],
    qmeta: dict[str, str],
    candidate: dict[str, Any],
    existing_questions_same_qtype: list[str] | None = None,
    generation_slot: int = 1,
) -> tuple[str, list[dict[str, Any]]]:
    retrieved_rows = candidate["retrieved_rows"]
    retrieved_facts = []
    for row in retrieved_rows:
        retrieved_facts.append(
            {
                "subject": norm_text(row.get("subject")),
                "relation": norm_text(row.get("relation")),
                "target": norm_text(row.get("target")),
                "via": norm_text(row.get("via")) or None,
                "hop": row.get("hop"),
                "score": row.get("score"),
                "verbalized_text": norm_text(row.get("verbalized_text")),
                "evidence": norm_text(row.get("evidence")),
                "source_url": norm_text(row.get("source_url")),
            }
        )

    prompt = INDIFOODVQA_PROMPT_TEMPLATE.format(
        image_description=image_row["image_description"],
        kg_triples=format_kg_triples_for_prompt(retrieved_rows),
        question_type=qmeta.get("question_type") or qmeta["canonical_qtype"],
        detailed_information=qmeta.get("detail_description", ""),
    )

    extras: list[str] = []
    if existing_questions_same_qtype:
        existing_text = "\n".join(f"- {q}" for q in existing_questions_same_qtype[:10])
        extras.append(
            "Avoid repeating the following existing questions for the same image and question type:\n"
            + existing_text
        )
    extras.append(f"Focus on producing diverse reasoning patterns. Generation slot: {generation_slot}.")
    if extras:
        prompt += "\n\n" + "\n\n".join(extras)

    return prompt, retrieved_facts


def call_gemini_generate(client, prompt_text: str, retries: int = 3) -> str | None:
    from google import genai

    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt_text,
                config=genai.types.GenerateContentConfig(
                    temperature=0.55,
                    max_output_tokens=4096,
                ),
            )
            raw = (resp.text or "").strip()
            if raw:
                return raw
        except Exception as exc:  # noqa: BLE001
            print(f"    Gemini attempt {attempt} failed: {type(exc).__name__}: {exc}")
            time.sleep(1.5 * attempt)
    return None


def parse_answer_choices_block(text: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
        m = re.match(r'^([ABCD])\s*[\.:\-)\]]?\s*(.+)$', line, flags=re.I)
        if not m:
            continue
        letter = m.group(1).upper()
        value = norm_text(m.group(2))
        if value:
            choices[letter] = value
    return choices


def parse_indifoodvqa_output(raw_text: str) -> list[dict[str, Any]]:
    if not raw_text:
        return []

    block_pattern = re.compile(
        r'Question\s*\n(?P<question>.+?)\n\s*\n\s*Answer Choices\s*\n(?P<choices>.+?)\n\s*\n\s*Correct Answer\s*\n(?P<answer>[ABCD])\s*\n\s*\n\s*Reason\s*\n(?P<reason>.*?)(?=\n\s*Question\s*\n|\Z)',
        flags=re.I | re.S,
    )
    items: list[dict[str, Any]] = []
    for match in block_pattern.finditer(raw_text.strip()):
        question = norm_text(match.group("question"))
        choices = parse_answer_choices_block(match.group("choices"))
        answer = norm_text(match.group("answer")).upper()
        reason = norm_text(match.group("reason"))
        items.append(
            {
                "question_vi": question,
                "choices": choices,
                "answer": answer,
                "rationale_vi": reason,
            }
        )
    return items


def validate_generation(result: dict[str, Any]) -> bool:
    if not result:
        return False
    choices = result.get("choices") or {}
    if set(choices.keys()) != {"A", "B", "C", "D"}:
        return False
    if result.get("answer") not in {"A", "B", "C", "D"}:
        return False
    if not norm_text(result.get("question_vi")):
        return False
    if not norm_text(result.get("rationale_vi")):
        return False
    if not all(norm_text(v) for v in choices.values()):
        return False
    return True


def generate_one_sample(
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
) -> dict[str, Any] | None:
    retrieval_query = build_retrieval_query(
        qmeta=qmeta,
        dish=anchor_dish,
        image_desc=image_row["image_description"],
        food_items=image_row["food_items"],
    )
    retrieved = kg.retrieve(items=[anchor_dish], question=retrieval_query, top_k=top_k)
    candidates = select_candidates(qmeta, retrieved)
    if not candidates:
        return None

    used_answer_keys = used_answer_keys or set()
    existing_questions_same_qtype = existing_questions_same_qtype or []

    ordered_candidates = [c for c in candidates if c.get("answer_key") not in used_answer_keys]
    if not ordered_candidates:
        ordered_candidates = candidates

    for candidate in ordered_candidates[:8]:
        prompt_text, retrieved_facts = build_indifoodvqa_prompt(
            image_row=image_row,
            qmeta=qmeta,
            candidate=candidate,
            existing_questions_same_qtype=existing_questions_same_qtype,
            generation_slot=generation_slot,
        )
        raw_text = call_gemini_generate(gemini_client, prompt_text)
        parsed_items = parse_indifoodvqa_output(raw_text or "")
        if not parsed_items:
            continue

        existing_norms = {norm_text(q).lower() for q in existing_questions_same_qtype}
        triples_used = shrink_triples(candidate["triples"])

        for llm_result in parsed_items:
            if not validate_generation(llm_result):
                continue
            question_norm = norm_text(llm_result.get("question_vi")).lower()
            if question_norm and question_norm in existing_norms:
                continue

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
            }

    return None


def question_signature(image_id: str, qtype: str, question_vi: str) -> str:
    return f"{image_id}::{qtype}::{slug(question_vi)}"


def build_existing_maps(samples: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], set[str]], set[str]]:
    questions_by_key: dict[tuple[str, str], list[str]] = {}
    answers_by_key: dict[tuple[str, str], set[str]] = {}
    question_keys: set[str] = set()
    for sample in samples:
        image_id = norm_text(sample.get("image_id"))
        qtype = norm_text(sample.get("qtype"))
        question_vi = norm_text(sample.get("question_vi"))
        answer_key = norm_text(sample.get("answer_key"))
        if not image_id or not qtype or not question_vi:
            continue
        pair_key = (image_id, qtype)
        questions_by_key.setdefault(pair_key, []).append(question_vi)
        if answer_key:
            answers_by_key.setdefault(pair_key, set()).add(answer_key)
        question_keys.add(question_signature(image_id, qtype, question_vi))
    return questions_by_key, answers_by_key, question_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ViFoodVQA samples using query.py retriever")
    parser.add_argument("--table", default="image")
    parser.add_argument("--id-col", default="image_id")
    parser.add_argument("--image-col", default="image_url")
    parser.add_argument("--desc-col", default="image_desc")
    parser.add_argument("--items-col", default="food_items")
    parser.add_argument("--question-types-csv", default=str(QUESTION_TYPES_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-image-id", default="")
    parser.add_argument("--end-image-id", default="")
    parser.add_argument("--image-ids-file", default="")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--limit-images", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--start-page", type=int, default=-1)
    parser.add_argument("--qtypes-per-image", type=int, default=0, help="0 = use all selected qtypes for each image")
    parser.add_argument("--qtypes", nargs="*", default=[])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--questions-per-qtype", type=int, default=1)
    parser.add_argument("--only-approved-images", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir, output_file, progress_file = resolve_output_paths(args.output_dir)

    all_qtypes = load_question_types(Path(args.question_types_csv))
    qmeta_by_key = {q["canonical_qtype"]: q for q in all_qtypes}

    requested_qtypes: list[dict[str, str]] = []
    if args.qtypes:
        for raw in args.qtypes:
            canon = QTYPE_ALIASES.get(norm_text(raw).lower(), norm_text(raw).lower())
            if canon not in qmeta_by_key:
                print(f"[WARN] Skip unsupported qtype: {raw} -> {canon}")
                continue
            requested_qtypes.append(qmeta_by_key[canon])
    else:
        requested_qtypes = list(qmeta_by_key.values())

    if not requested_qtypes:
        print("No supported question types selected.")
        sys.exit(1)

    progress = load_progress(progress_file)
    limit_samples = args.limit_samples or args.limit_images
    if args.start_page >= 0:
        progress["page"] = args.start_page

    generated = list(progress.get("generated", []))
    questions_by_pair, answers_by_pair, question_keys = build_existing_maps(generated)

    allowed_image_ids = load_allowed_image_ids(args.image_ids_file)

    supabase = make_supabase_client()
    gemini_client = make_gemini_client()
    kg = make_retriever(device=args.device)

    try:
        embedded_edges = count_embedded_edges(kg)
        if embedded_edges == 0:
            raise RuntimeError(
                "No relationship embeddings found in Neo4j. Run src/05_kg_vectorizer.py first."
            )

        substitution_emb = count_substitution_embeddings(kg)
        if substitution_emb == 0 and any(q["canonical_qtype"] == "substitution_rules" for q in requested_qtypes):
            print(
                "[WARN] substitution_rules is disabled in practice because current edge embeddings for "
                "fromIngredient/toIngredient are missing. Skip this qtype for now or update 05_kg_vectorizer.py."
            )
            requested_qtypes = [q for q in requested_qtypes if q["canonical_qtype"] != "substitution_rules"]

        dish_aliases = fetch_all_dishes(kg)
        print(f"Loaded {len(dish_aliases)} dishes from Neo4j")
        print(f"Embedded relationships: {embedded_edges}")
        print("Question types:", ", ".join(q["canonical_qtype"] for q in requested_qtypes))
        if norm_text(args.start_image_id) or norm_text(args.end_image_id):
            print(
                "Image range:",
                norm_text(args.start_image_id) or "<min>",
                "->",
                norm_text(args.end_image_id) or "<max>",
            )
        if allowed_image_ids:
            print(f"Filtered image ids: {len(allowed_image_ids)}")
        if limit_samples > 0:
            print(f"Limit samples: {limit_samples}")
        print(f"Questions per qtype: {args.questions_per_qtype}")

        page = progress["page"]
        attempted_images = 0

        while True:
            rows = fetch_image_rows(
                client=supabase,
                table=args.table,
                id_col=args.id_col,
                image_col=args.image_col,
                desc_col=args.desc_col,
                items_col=args.items_col,
                page=page,
                size=PAGE_SIZE,
                only_approved_images=args.only_approved_images,
                start_image_id=norm_text(args.start_image_id),
                end_image_id=norm_text(args.end_image_id),
                allowed_image_ids=allowed_image_ids,
            )
            if not rows:
                break

            print(f"\nPage {page}: {len(rows)} image rows")
            for raw in rows:
                if limit_samples > 0 and len(generated) >= limit_samples:
                    break

                image_row = {
                    "image_id": norm_text(raw.get(args.id_col)),
                    "image": norm_text(raw.get(args.image_col)),
                    "image_description": norm_text(raw.get(args.desc_col)),
                    "food_items": [norm_text(x) for x in (raw.get(args.items_col) or []) if norm_text(x)],
                }
                attempted_images += 1

                if not image_row["image_id"] or not image_row["image_description"] or not image_row["food_items"]:
                    continue

                anchor_dish = choose_anchor_dish(image_row["food_items"], dish_aliases)
                if not anchor_dish:
                    continue

                qtypes_for_image = requested_qtypes[:]
                rng.shuffle(qtypes_for_image)
                if args.qtypes_per_image and args.qtypes_per_image > 0:
                    qtypes_for_image = qtypes_for_image[: args.qtypes_per_image]

                for qmeta in qtypes_for_image:
                    if limit_samples > 0 and len(generated) >= limit_samples:
                        break

                    pair_key = (image_row["image_id"], qmeta["canonical_qtype"])
                    existing_questions = questions_by_pair.get(pair_key, [])
                    used_answer_keys = answers_by_pair.get(pair_key, set())

                    while len(existing_questions) < args.questions_per_qtype:
                        if limit_samples > 0 and len(generated) >= limit_samples:
                            break

                        sample = generate_one_sample(
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
                        if not sample:
                            break

                        qsig = question_signature(sample["image_id"], sample["qtype"], sample["question_vi"])
                        if qsig in question_keys:
                            break

                        generated.append(sample)
                        question_keys.add(qsig)
                        existing_questions = questions_by_pair.setdefault(pair_key, [])
                        existing_questions.append(sample["question_vi"])
                        if norm_text(sample.get("answer_key")):
                            answers_by_pair.setdefault(pair_key, set()).add(norm_text(sample.get("answer_key")))
                            used_answer_keys = answers_by_pair[pair_key]
                        print(
                            f"  ✓ {image_row['image_id']}::{qmeta['canonical_qtype']}::{len(existing_questions)} "
                            f"-> {sample['answer']} | {sample['question_vi'][:90]}"
                        )

                        progress = {
                            "page": page,
                            "generated": generated,
                            "question_keys": sorted(question_keys),
                        }
                        save_progress(progress, output_dir, progress_file)

            page += 1
            progress["page"] = page
            save_progress(progress, output_dir, progress_file)

            if limit_samples > 0 and len(generated) >= limit_samples:
                break

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n-- Done --")
        print(f"Attempted images: {attempted_images}")
        print(f"Generated samples: {len(generated)}")
        print(f"Saved: {output_file}")
        print(f"Checkpoint: {progress_file}")

    finally:
        kg.close()


if __name__ == "__main__":
    main()
