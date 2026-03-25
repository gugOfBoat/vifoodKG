"""
ViFoodKG — Step 4: Knowledge Graph Vectorizer
==============================================
Embeds every relationship's `verbalized_text` using
`intfloat/multilingual-e5-small` and stores the vector as
`embedding` property on each edge.

After embedding, creates a Neo4j Vector Index for cosine
similarity search over the relationship embeddings.

Usage
-----
  python src/05_kg_vectorizer.py                  # embed all un-embedded edges
  python src/05_kg_vectorizer.py --batch-size 64  # larger batches (needs more RAM)
  python src/05_kg_vectorizer.py --dry-run        # count edges only, no write
"""

import argparse
import os
import sys
from typing import Generator

from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME  = "intfloat/multilingual-e5-small"
VECTOR_DIM  = 384
E5_PREFIX   = "passage: "   # required prefix for retrieval passages

# Relation types we want to vectorize (excluding structural/reification edges)
VECTORIZE_REL_TYPES = [
    "hasIngredient", "servedWith", "originRegion", "dishType",
    "cookingTechnique", "flavorProfile", "hasAllergen", "hasDietaryTag",
    "ingredientCategory", "hasSubRule", "fromIngredient", "toIngredient"
]


# ══════════════════════════════════════════════════════════════════════════════
# Neo4j helpers
# ══════════════════════════════════════════════════════════════════════════════

FETCH_QUERY = """
MATCH ()-[r]-()
WHERE type(r) IN $rel_types
  AND r.verbalized_text IS NOT NULL
  AND r.embedding IS NULL
RETURN elementId(r) AS eid, r.verbalized_text AS text
"""

UPDATE_QUERY = """
UNWIND $rows AS row
MATCH ()-[r]-() WHERE elementId(r) = row.eid
SET r.embedding = row.embedding
"""

VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX triple_vector_index IF NOT EXISTS
FOR ()-[r:hasIngredient|servedWith|originRegion|dishType|cookingTechnique|flavorProfile|hasAllergen|hasDietaryTag|ingredientCategory|hasSubRule|fromIngredient|toIngredient]-()
ON (r.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}}
"""


def fetch_unembedded(session, rel_types: list) -> list[dict]:
    """Return list of {eid, text} for all unembedded relations."""
    result = session.run(FETCH_QUERY, rel_types=rel_types)
    return [{"eid": r["eid"], "text": r["text"]} for r in result]


def bulk_update(session, rows: list[dict]) -> None:
    """Write embedding vectors back to Neo4j via UNWIND."""
    session.run(UPDATE_QUERY, rows=rows)


def create_vector_index(session) -> None:
    """Drop old index and recreate cosine vector index on all relationship embeddings."""
    try:
        session.run("DROP INDEX triple_vector_index IF EXISTS")
        print("  Dropped old `triple_vector_index` (if existed).")
        session.run(VECTOR_INDEX_QUERY)
        print("  Vector index `triple_vector_index` created successfully.")
    except Exception as e:
        print(f"  [WARN] Vector index creation: {e}")


def batched(lst: list, size: int) -> Generator:
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


# ══════════════════════════════════════════════════════════════════════════════
# Embedding
# ══════════════════════════════════════════════════════════════════════════════

def load_model(device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        sys.exit(1)
    print(f"Loading model `{MODEL_NAME}` on device `{device}`...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("  Model loaded.\n")
    return model


def embed_batch(model, texts: list[str]) -> list[list[float]]:
    """Embed list of texts; returns list of float vectors."""
    prefixed = [E5_PREFIX + t for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(driver, batch_size: int, device: str, dry_run: bool) -> None:
    with driver.session() as session:
        print("Fetching unembedded relationships from Neo4j...")
        rows = fetch_unembedded(session, VECTORIZE_REL_TYPES)
        total = len(rows)
        print(f"  Found {total} unembedded relationship(s).\n")

        if dry_run or total == 0:
            if dry_run:
                print("✓ Dry-run mode — no embeddings written.")
            else:
                print("✓ Nothing to embed. Vector index will still be created.")
            create_vector_index(session)
            return

        model = load_model(device)

        done = 0
        for batch in batched(rows, batch_size):
            texts    = [r["text"] for r in batch]
            vectors  = embed_batch(model, texts)

            write_rows = [
                {"eid": r["eid"], "embedding": vec}
                for r, vec in zip(batch, vectors)
            ]
            bulk_update(session, write_rows)

            done += len(batch)
            pct = done / total * 100
            print(f"  Embedded {done:>5}/{total} ({pct:5.1f}%) relationships...")

        print(f"\n✓ Embedding complete. {done} relationships updated.\n")
        print("Creating vector index...")
        create_vector_index(session)
        print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="ViFoodKG Step 4 — Edge Vectorizer")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of texts per embedding batch (default: 32)")
    parser.add_argument("--device", default="cpu",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for SentenceTransformer (default: cpu)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count edges only, do not write embeddings")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

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
        run(driver, batch_size=args.batch_size, device=device, dry_run=args.dry_run)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
