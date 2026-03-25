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

def run_ingestion_bulk(tx, data: list):
    from collections import defaultdict
    
    # 1. Ensure all Dish nodes exist
    dishes = [{"name": d.get("dish")} for d in data if d.get("dish")]
    tx.run("UNWIND $dishes AS d MERGE (n:Dish {name: d.name}) SET n.source = 'ViFoodKG'", dishes=dishes)
    
    normal_groups = defaultdict(list)
    subrule_groups = []
    fromto_groups = defaultdict(list)
    
    for d in data:
        dish_name = d.get("dish")
        if not dish_name: continue
        for triple in d.get("triples", []):
            rel = triple.get("relation", "")
            tgt = triple.get("target", "")
            ttype = canon_label(triple.get("target_type", "Entity"))
            subj = triple.get("subject", dish_name)
            s_url = triple.get("source_url", "")
            evid = triple.get("evidence", "")
            
            if not rel or not tgt: continue
            
            if rel == "hasSubRule":
                subrule_groups.append({
                    "dish": dish_name, "rule": tgt, "u": s_url, "e": evid, 
                    "v": f"{dish_name} có quy tắc thay thế {tgt}"
                })
            elif rel in ("fromIngredient", "toIngredient"):
                vt = f"{subj} thay thế nguyên liệu {tgt}" if rel == "fromIngredient" else f"{subj} bằng {tgt}"
                fromto_groups[rel].append({
                    "rule": subj, "ing": tgt, "u": s_url, "e": evid, "v": vt
                })
            else:
                s_label = "Dish" if subj == dish_name else "Ingredient"
                vn_verb = VERBALIZATION.get(rel, rel)
                normal_groups[(s_label, rel, ttype)].append({
                    "s": subj, "t": tgt, "u": s_url, "e": evid, "v": f"{subj} {vn_verb} {tgt}"
                })
                
    # 2. Insert normal groups
    for (s_label, rel, t_label), items in normal_groups.items():
        q = f"""
        UNWIND $items AS item
        MERGE (s:{s_label} {{name: item.s}})
        MERGE (t:{t_label} {{name: item.t}})
        MERGE (s)-[r:{rel}]->(t)
        SET r.source_url = item.u, r.evidence = item.e, r.verbalized_text = item.v
        """
        tx.run(q, items=items)
        
    # 3. Insert hasSubRule
    if subrule_groups:
        q_sr = """
        UNWIND $items AS item
        MERGE (d:Dish {name: item.dish})
        MERGE (sr:SubstitutionRule {name: item.rule})
        MERGE (d)-[r:hasSubRule]->(sr)
        SET r.source_url = item.u, r.evidence = item.e, r.verbalized_text = item.v
        """
        tx.run(q_sr, items=subrule_groups)
        
    # 4. Insert from/toIngredient
    for rel, items in fromto_groups.items():
        q_ft = f"""
        UNWIND $items AS item
        MERGE (sr:SubstitutionRule {{name: item.rule}})
        MERGE (ing:Ingredient {{name: item.ing}})
        MERGE (sr)-[r:{rel}]->(ing)
        SET r.source_url = item.u, r.evidence = item.e, r.verbalized_text = item.v
        """
        tx.run(q_ft, items=items)


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

        def write_chunk(tx):
            run_ingestion_bulk(tx, data)
            
        print("  Writing all records to Neo4j via Bulk UNWIND...")
        session.execute_write(write_chunk)

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
