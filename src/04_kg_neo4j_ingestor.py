"""
ViFoodKG — Step 3: Neo4j Knowledge Graph Ingestor
==================================================
Loads extracted triples from data/triples/extracted_triples.json (or
_progress.json if the full extract is not yet done) into Neo4j Cloud.

Schema
------
  Nodes  : Dish, Ingredient, IngredientCategory, Region, DishType,
            CookingTechnique, FlavorProfile, DietaryTag, Allergen,
            SideDish, Condiment, SubstitutionRule
  Edges  : hasIngredient, servedWith, originRegion, dishType,
            cookingTechnique, flavorProfile, ingredientCategory,
            hasAllergen, hasDietaryTag, hasSubRule,
            fromIngredient, toIngredient

Every edge carries:  source_url, evidence, verbalized_text

Usage
-----
  python src/04_kg_neo4j_ingestor.py              # load extracted_triples.json
  python src/04_kg_neo4j_ingestor.py --dry-run    # parse only, do not write
  python src/04_kg_neo4j_ingestor.py --source progress  # use _progress.json
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
TRIPLES_FILE    = PROJECT_ROOT / "data" / "triples" / "extracted_triples.json"
PROGRESS_FILE   = PROJECT_ROOT / "data" / "triples" / "_progress.json"
MASTER_FILE     = PROJECT_ROOT / "data" / "master_entities.json"

# ── Verbalization map ─────────────────────────────────────────────────────────
VERBALIZATION: dict[str, str] = {
    "hasIngredient":    "có nguyên liệu là",
    "servedWith":       "thường được ăn kèm với",
    "originRegion":     "có nguồn gốc từ vùng",
    "dishType":         "là loại món ăn",
    "cookingTechnique": "được chế biến bằng kỹ thuật",
    "flavorProfile":    "có đặc trưng hương vị là",
    "hasAllergen":      "có chứa chất gây dị ứng là",
    "hasDietaryTag":    "phù hợp với chế độ ăn",
    "ingredientCategory": "thuộc nhóm nguyên liệu",
    # SubstitutionRule edges do not get verbalized on the edge level
}

# ── target_type → Neo4j Label normalizer ─────────────────────────────────────
# Some LLM outputs use "SideDish/Condiment" — split to first token
LABEL_CANON: dict[str, str] = {
    "Dish":               "Dish",
    "Ingredient":         "Ingredient",
    "IngredientCategory": "IngredientCategory",
    "Region":             "Region",
    "DishType":           "DishType",
    "CookingTechnique":   "CookingTechnique",
    "FlavorProfile":      "FlavorProfile",
    "DietaryTag":         "DietaryTag",
    "Allergen":           "Allergen",
    "SideDish":           "SideDish",
    "Condiment":          "Condiment",
    "SubstitutionRule":   "SubstitutionRule",
    "SideDish/Condiment": "Condiment",   # common LLM variant
}


def canon_label(raw: str) -> str:
    """Return the canonical Neo4j node label for a target_type string."""
    raw = (raw or "").strip()
    return LABEL_CANON.get(raw, raw.split("/")[0].strip() or "Entity")


# ══════════════════════════════════════════════════════════════════════════════
# Cypher helpers
# ══════════════════════════════════════════════════════════════════════════════

def create_indexes(session) -> None:
    """Create indexes for fast MERGE on all node labels."""
    labels = [
        "Dish", "Ingredient", "IngredientCategory", "Region", "DishType",
        "CookingTechnique", "FlavorProfile", "DietaryTag", "Allergen",
        "SideDish", "Condiment", "SubstitutionRule",
    ]
    for label in labels:
        try:
            session.run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)"
            )
        except Exception as e:
            print(f"  [WARN] Index for :{label} → {e}")
    print("  Indexes ready.")


def merge_node(tx, label: str, name: str, extra: dict | None = None) -> None:
    """MERGE a node with given label and name; optionally set extra props."""
    props = {"name": name}
    if extra:
        props.update(extra)
    cypher = (
        f"MERGE (n:{label} {{name: $name}}) "
        "SET n += $props"
    )
    tx.run(cypher, name=name, props=props)


def merge_edge(
    tx,
    subj_label: str,
    subj_name: str,
    rel: str,
    obj_label: str,
    obj_name: str,
    source_url: str,
    evidence: str,
    verbalized: str = "",
) -> None:
    """MERGE both nodes and the relationship between them."""
    cypher = (
        f"MERGE (a:{subj_label} {{name: $subj}}) "
        f"MERGE (b:{obj_label} {{name: $obj}}) "
        f"MERGE (a)-[r:{rel}]->(b) "
        "SET r.source_url = $source_url, "
        "    r.evidence   = $evidence, "
        "    r.verbalized_text = $verbalized"
    )
    tx.run(
        cypher,
        subj=subj_name,
        obj=obj_name,
        source_url=source_url,
        evidence=evidence,
        verbalized=verbalized,
    )


def merge_substitution_rule(
    tx,
    dish_name: str,
    rule_name: str,
    from_ingredient: str,
    to_ingredient: str,
    source_url: str,
    evidence: str,
) -> None:
    """
    Reification pattern:
        (Dish)-[:hasSubRule]->(SubstitutionRule)
        (SubstitutionRule)-[:fromIngredient]->(Ingredient A)
        (SubstitutionRule)-[:toIngredient]->(Ingredient B)
    """
    cypher = """
    MERGE (d:Dish {name: $dish})
    MERGE (sr:SubstitutionRule {name: $rule_name})
    MERGE (fa:Ingredient {name: $from_ingredient})
    MERGE (ta:Ingredient {name: $to_ingredient})
    MERGE (d)-[r1:hasSubRule]->(sr)
      SET r1.source_url = $source_url, r1.evidence = $evidence,
          r1.verbalized_text = $dish + ' có quy tắc thay thế: ' + $rule_name
    MERGE (sr)-[r2:fromIngredient]->(fa)
      SET r2.source_url = $source_url, r2.evidence = $evidence,
          r2.verbalized_text = $rule_name + ' thay thế nguyên liệu ' + $from_ingredient
    MERGE (sr)-[r3:toIngredient]->(ta)
      SET r3.source_url = $source_url, r3.evidence = $evidence,
          r3.verbalized_text = $rule_name + ' bằng ' + $to_ingredient
    """
    tx.run(
        cypher,
        dish=dish_name,
        rule_name=rule_name,
        from_ingredient=from_ingredient,
        to_ingredient=to_ingredient,
        source_url=source_url,
        evidence=evidence,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Ingest logic
# ══════════════════════════════════════════════════════════════════════════════

def ingest_triple(tx, dish_name: str, triple: dict) -> None:
    """Process a single triple and write to Neo4j."""
    relation   = triple.get("relation", "")
    target     = triple.get("target", "")
    ttype_raw  = triple.get("target_type", "Entity")
    subject    = triple.get("subject", dish_name)   # some triples override subject
    source_url = triple.get("source_url", "")
    evidence   = triple.get("evidence", "")
    obj_label  = canon_label(ttype_raw)

    if not relation or not target:
        return

    # ── Special: SubstitutionRule ──────────────────────────────────────────
    if relation == "hasSubRule":
        # `target` is the SubstitutionRule node name (generated by LLM)
        # from/toIngredient triples will come as separate items; we just
        # ensure the Dish→SubstitutionRule edge exists here.
        merge_node(tx, "SubstitutionRule", target)
        merge_edge(
            tx,
            subj_label="Dish",
            subj_name=dish_name,
            rel="hasSubRule",
            obj_label="SubstitutionRule",
            obj_name=target,
            source_url=source_url,
            evidence=evidence,
            verbalized=f"{dish_name} có quy tắc thay thế {target}",
        )
        return

    if relation in ("fromIngredient", "toIngredient"):
        # subject here is the SubstitutionRule node
        merge_node(tx, "Ingredient", target)
        merge_node(tx, "SubstitutionRule", subject)
        cypher = (
            f"MERGE (sr:SubstitutionRule {{name: $subj}}) "
            f"MERGE (ing:Ingredient {{name: $target}}) "
            f"MERGE (sr)-[r:{relation}]->(ing) "
            "SET r.source_url = $source_url, r.evidence = $evidence, "
            "    r.verbalized_text = $vt"
        )
        if relation == "fromIngredient":
            vt = f"{subject} thay thế nguyên liệu {target}"
        else:
            vt = f"{subject} bằng {target}"
        tx.run(cypher, subj=subject, target=target,
               source_url=source_url, evidence=evidence, vt=vt)
        return

    # ── Normal edge ────────────────────────────────────────────────────────
    # Determine subject label: if subject == dish_name it's a Dish; else guess
    subj_label = "Dish" if subject == dish_name else "Ingredient"

    vn_verb = VERBALIZATION.get(relation, relation)
    verbalized = f"{subject} {vn_verb} {target}"

    merge_edge(
        tx,
        subj_label=subj_label,
        subj_name=subject,
        rel=relation,
        obj_label=obj_label,
        obj_name=target,
        source_url=source_url,
        evidence=evidence,
        verbalized=verbalized,
    )


def run_ingestion(data: list, driver, dry_run: bool = False) -> None:
    """Main ingestion loop."""
    total_dishes  = len(data)
    total_triples = sum(len(d.get("triples", [])) for d in data)
    print(f"\n→ Ingest plan: {total_dishes} dishes | {total_triples} triples | dry_run={dry_run}\n")

    if dry_run:
        # Just show the plan — no DB calls
        for i, dish_record in enumerate(data[:20], 1):
            dish_name = dish_record.get("dish", "")
            triples   = dish_record.get("triples", [])
            print(f"  [{i:>4}/{total_dishes}] {dish_name}  — {len(triples)} triples")
        if total_dishes > 20:
            print(f"  ... ({total_dishes - 20} more dishes not shown)")
        print(f"\n✓ Dry-run OK — {total_dishes} Dish records ready. Run without --dry-run to write to Neo4j.")
        return

    with driver.session() as session:
        print("Creating indexes...")
        create_indexes(session)

        for i, dish_record in enumerate(data, 1):
            dish_name = dish_record.get("dish", "")
            triples   = dish_record.get("triples", [])

            if not dish_name:
                continue

            pct = i / total_dishes * 100
            print(f"  [{i:>4}/{total_dishes}] ({pct:5.1f}%) {dish_name}  — {len(triples)} triples")

            def write_block(tx, dish_name=dish_name, triples=triples):
                merge_node(tx, "Dish", dish_name, {"source": "ViFoodKG"})
                for triple in triples:
                    try:
                        ingest_triple(tx, dish_name, triple)
                    except Exception as e:
                        print(f"      [SKIP triple] {triple} → {e}")

            session.execute_write(write_block)

    print(f"\n✓ Ingestion complete. {total_dishes} Dish nodes processed.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ViFoodKG Step 3 - Neo4j Ingestor")
    parser.add_argument(
        "--source", choices=["extracted", "progress"], default="extracted",
        help="Use extracted_triples.json (default) or _progress.json"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and print plan, do not write to Neo4j")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    if args.source == "progress":
        src_file = PROGRESS_FILE
    else:
        src_file = TRIPLES_FILE

    if not src_file.exists():
        print(f"[ERROR] File not found: {src_file}")
        print("  Hint: run src/03_kg_triple_extractor.py first, or use --source progress")
        sys.exit(1)

    raw = json.loads(src_file.read_text(encoding="utf-8"))
    if isinstance(raw, dict):               # _progress.json wrapper
        data = raw.get("all_triples", [])
    else:
        data = raw

    print(f"Loaded {len(data)} dish records from {src_file.name}")

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        run_ingestion(data, driver=None, dry_run=True)
        return

    # ── Connect to Neo4j ──────────────────────────────────────────────────
    if not NEO4J_PASSWORD:
        print("[ERROR] NEO4J_PASSWORD not set in .env")
        sys.exit(1)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[ERROR] neo4j driver not installed. Run: pip install neo4j")
        sys.exit(1)

    print(f"Connecting to Neo4j: {NEO4J_URI}  (user: {NEO4J_USER})")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("  Connection OK\n")
        run_ingestion(data, driver, dry_run=False)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
