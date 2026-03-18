# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ViFoodKG — Step 4: Edge Vectorizer (Google Colab T4 Version)            ║
# ║  Run each cell top-to-bottom on a T4 GPU runtime.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── CELL 1: Install dependencies ─────────────────────────────────────────────
# @title Cell 1: Install

import subprocess
subprocess.run(["pip", "install", "-q", "neo4j", "sentence-transformers"], check=True)
print("✓ Dependencies installed")

# ── CELL 2: Configuration — fill in your credentials ─────────────────────────
# @title Cell 2: Configuration

NEO4J_URI      = "neo4j+s://aa4eacb4.databases.neo4j.io"   # ← Aura URI
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "YOUR_PASSWORD_HERE"                  # ← Aura password

MODEL_NAME  = "intfloat/multilingual-e5-small"
BATCH_SIZE  = 64      # T4 can handle larger batches
VECTOR_DIM  = 384
E5_PREFIX   = "passage: "

VECTORIZE_REL_TYPES = [
    "hasIngredient", "servedWith", "originRegion", "dishType",
    "cookingTechnique", "flavorProfile", "hasAllergen", "hasDietaryTag",
    "ingredientCategory",
]

# ── CELL 3: Verify GPU ────────────────────────────────────────────────────────
# @title Cell 3: Check GPU

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM available: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
else:
    print("⚠ GPU not found — running on CPU (slower)")

# ── CELL 4: Connect to Neo4j ──────────────────────────────────────────────────
# @title Cell 4: Connect to Neo4j

from neo4j import GraphDatabase

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
driver.verify_connectivity()
print(f"✓ Connected to Neo4j: {NEO4J_URI}")

# ── CELL 5: Fetch unembedded relationships ────────────────────────────────────
# @title Cell 5: Fetch unembedded edges

FETCH_QUERY = """
MATCH ()-[r]-()
WHERE type(r) IN $rel_types
  AND r.verbalized_text IS NOT NULL
  AND r.embedding IS NULL
RETURN elementId(r) AS eid, r.verbalized_text AS text
"""

with driver.session() as session:
    result = session.run(FETCH_QUERY, rel_types=VECTORIZE_REL_TYPES)
    rows = [{"eid": r["eid"], "text": r["text"]} for r in result]

total = len(rows)
print(f"Found {total} unembedded relationship(s) to process.")
if rows:
    print(f"\nExample:")
    print(f"  eid  : {rows[0]['eid']}")
    print(f"  text : {rows[0]['text']}")

# ── CELL 6: Load embedding model ──────────────────────────────────────────────
# @title Cell 6: Load E5 model

from sentence_transformers import SentenceTransformer

print(f"Loading {MODEL_NAME} on {device}...")
model = SentenceTransformer(MODEL_NAME, device=device)
print("✓ Model loaded")

# Quick smoke test
test_vec = model.encode(["passage: test"], normalize_embeddings=True)
print(f"  Output dimension: {test_vec.shape[1]}")

# ── CELL 7: Embed + Upload ────────────────────────────────────────────────────
# @title Cell 7: Embed and upload to Neo4j

UPDATE_QUERY = """
UNWIND $rows AS row
MATCH ()-[r]-() WHERE elementId(r) = row.eid
SET r.embedding = row.embedding
"""

def batched(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i: i + size]

done = 0
with driver.session() as session:
    for batch in batched(rows, BATCH_SIZE):
        texts   = [E5_PREFIX + r["text"] for r in batch]
        vectors = model.encode(texts, normalize_embeddings=True,
                               show_progress_bar=False)

        write_rows = [
            {"eid": r["eid"], "embedding": v.tolist()}
            for r, v in zip(batch, vectors)
        ]
        session.run(UPDATE_QUERY, rows=write_rows)

        done += len(batch)
        print(f"  Embedded {done:>5}/{total} ({done/total*100:5.1f}%) ...")

print(f"\n✓ Done! {done} relationships now have `embedding` property.")

# ── CELL 8: Create Vector Index ───────────────────────────────────────────────
# @title Cell 8: Create Vector Index

VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX triple_vector_index IF NOT EXISTS
FOR ()-[r:hasIngredient|servedWith|originRegion|dishType|cookingTechnique|flavorProfile|hasAllergen|hasDietaryTag|ingredientCategory]-()
ON (r.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function সিঙ্গ': 'cosine'
}}
"""

with driver.session() as session:
    try:
        session.run(VECTOR_INDEX_QUERY)
        print("✓ Vector index `triple_vector_index` created / already exists.")
    except Exception as e:
        print(f"[WARN] {e}")

driver.close()
print("✓ Neo4j driver closed.")
