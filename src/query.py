"""
ViFoodKG — KGRetriever Module
===============================
Chiến lược: Neo → Traverse → Rank

  1. Neo   : Anchor vào đúng các Dish node trong items
  2. Traverse: Lấy toàn bộ edges 1-hop và 2-hop từ những node đó
  3. Rank  : Tính cosine giữa embedding của edges và vector câu hỏi → top-K

Ưu điểm:
- Load model 1 lần duy nhất (singleton), gọi lại .retrieve() bao nhiêu cũng được
- Không scan toàn bộ vector index → nhanh hơn, chính xác hơn
- Hỗ trợ cả 1-hop lẫn 2-hop tự động
- Dễ import từ Colab / pipeline sinh QA

Usage:
------
  from src.query import KGRetriever

  kg = KGRetriever()                         # load model 1 lần
  results = kg.retrieve(
      items=["Phở Bò", "Thịt Bò"],
      question="nguyên liệu chính là gì",
      top_k=5,
  )
  for r in results:
      print(r["verbalized_text"], r["score"])

CLI:
----
  python src/query.py -i "Phở Bò" "Thịt Bò" -q "nguyên liệu" -k 5
  python src/query.py -i "Bánh Xèo" -q "chất dị ứng" -k 3 --json
"""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

MODEL_NAME = "intfloat/multilingual-e5-small"
E5_QUERY_PREFIX = "query: "


# ══════════════════════════════════════════════════════════════════════════════
# CYPHER: Traverse neighborhood (1-hop + 2-hop)
# ══════════════════════════════════════════════════════════════════════════════

_TRAVERSE_QUERY = """
// ── 1-hop: Dish → direct target ─────────────────────────────────────────────
MATCH (d:Dish)-[r]->(target)
WHERE d.name IN $items
  AND NOT type(r) IN ['hasSubRule']      // SubRule handled separately
  AND r.embedding IS NOT NULL

RETURN
  d.name          AS subject,
  'Dish'          AS subject_type,
  type(r)         AS relation,
  target.name     AS target,
  labels(target)[0] AS target_type,
  null            AS via,               // no intermediate node for 1-hop
  r.verbalized_text AS verbalized_text,
  r.evidence      AS evidence,
  r.source_url    AS source_url,
  1               AS hop,
  r.embedding     AS embedding

UNION ALL

// ── 2-hop: Dish → Ingredient → (Category | Allergen | DietaryTag) ───────────
MATCH (d:Dish)-[:hasIngredient]->(ing:Ingredient)-[r]->(target)
WHERE d.name IN $items
  AND r.embedding IS NOT NULL

RETURN
  d.name          AS subject,
  'Dish'          AS subject_type,
  type(r)         AS relation,
  target.name     AS target,
  labels(target)[0] AS target_type,
  ing.name        AS via,
  r.verbalized_text AS verbalized_text,
  r.evidence      AS evidence,
  r.source_url    AS source_url,
  2               AS hop,
  r.embedding     AS embedding

UNION ALL

// ── SubstitutionRule: Dish → SubRule → from/to Ingredient ───────────────────
MATCH (d:Dish)-[:hasSubRule]->(sr:SubstitutionRule)-[r]->(ing:Ingredient)
WHERE d.name IN $items
  AND r.embedding IS NOT NULL

RETURN
  d.name          AS subject,
  'Dish'          AS subject_type,
  type(r)         AS relation,
  ing.name        AS target,
  'Ingredient'    AS target_type,
  sr.name         AS via,
  r.verbalized_text AS verbalized_text,
  r.evidence      AS evidence,
  r.source_url    AS source_url,
  2               AS hop,
  r.embedding     AS embedding
"""


# ══════════════════════════════════════════════════════════════════════════════
# KGRetriever — singleton class
# ══════════════════════════════════════════════════════════════════════════════

class KGRetriever:
    """
    Load model ONCE, call .retrieve() as many times as needed.
    Thread-safe for sequential calls (no shared mutable state between calls).
    """

    def __init__(
        self,
        device: str = "auto",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        # ── Neo4j driver ─────────────────────────────────────────────────────
        from neo4j import GraphDatabase
        uri  = neo4j_uri      or NEO4J_URI
        user = neo4j_user     or NEO4J_USER
        pw   = neo4j_password or NEO4J_PASSWORD

        if not pw:
            raise ValueError("NEO4J_PASSWORD not set (env or constructor param)")

        self._driver = GraphDatabase.driver(uri, auth=(user, pw))
        self._driver.verify_connectivity()

        # ── Embedding model ───────────────────────────────────────────────────
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        print(f"[KGRetriever] Loading {MODEL_NAME} on {device}...")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(MODEL_NAME, device=device)
        print("[KGRetriever] Ready.\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [E5_QUERY_PREFIX + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0]

    def _cosine(self, vec_a: np.ndarray, vec_b: list) -> float:
        b = np.array(vec_b, dtype=np.float32)
        # Both are L2-normalised → dot product == cosine
        return float(np.dot(vec_a, b))

    # ── Main API ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        items: list[str],
        question: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Parameters
        ----------
        items    : food entities detected in image (dish names, ingredient names)
        question : free-text description of what the user wants to know
        top_k    : number of top triples to return

        Returns
        -------
        List[dict] sorted by score desc, each with:
          subject, subject_type, relation, target, target_type,
          via (intermediate node for 2-hop, else None),
          verbalized_text, evidence, source_url, hop, score
        """
        # Step 1 — Embed question
        query_vec = self._embed(question)

        # Step 2 — Traverse neighborhood in Neo4j (anchor on items)
        with self._driver.session() as session:
            raw = session.run(_TRAVERSE_QUERY, items=items)
            rows = [dict(r) for r in raw]

        if not rows:
            return []

        # Step 3 — Rank by cosine, keep top-K
        scored = []
        for row in rows:
            emb = row.pop("embedding")
            if emb is None:
                continue
            score = self._cosine(query_vec, emb)
            scored.append({**row, "score": round(score, 6)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def retrieve_all_types(
        self,
        items: list[str],
        questions: list[str],
        top_k: int = 5,
    ) -> dict[str, list[dict]]:
        """
        Run retrieve() for a list of questions without reloading the model.
        Returns dict mapping each question → list of triples.

        Useful for iterating over all Question Types in one session.
        """
        return {q: self.retrieve(items, q, top_k) for q in questions}

    def close(self):
        self._driver.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printer
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: list[dict], items: list[str], question: str) -> None:
    print(f"\n{'═'*72}")
    print(f"  Items    : {', '.join(items)}")
    print(f"  Question : {question}")
    print(f"  Results  : {len(results)} triples")
    print(f"{'═'*72}\n")

    if not results:
        print("  (Không tìm thấy kết quả)\n")
        return

    for i, r in enumerate(results, 1):
        hop  = r.get("hop", "?")
        via  = r.get("via")
        subj = r.get("subject", "?")
        rel  = r.get("relation", "?")
        tgt  = r.get("target", "?")
        st   = r.get("subject_type", "")
        tt   = r.get("target_type", "")
        score = r.get("score", 0)

        print(f"  #{i}  [{hop}-hop]  score={score:.4f}")
        if via:
            print(f"    ({st}) {subj} → ({rel}) via [{via}] → {tgt} ({tt})")
        else:
            print(f"    ({st}) {subj} ──[{rel}]──▶ {tgt} ({tt})")

        vt = r.get("verbalized_text")
        if vt:
            print(f"    ✎ {vt}")
        ev = r.get("evidence")
        if ev:
            print(f"    📖 {ev[:140]}{'...' if len(ev) > 140 else ''}")
        src = r.get("source_url")
        if src and src != "LLM_Knowledge":
            print(f"    🔗 {src}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ViFoodKG — Neo→Traverse→Rank retriever",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--items", "-i", nargs="+", required=True, metavar="ITEM")
    parser.add_argument("--question", "-q", required=True, metavar="QUESTION")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with KGRetriever(device=args.device) as kg:
        results = kg.retrieve(args.items, args.question, args.top_k)

    if args.json:
        print(json.dumps({
            "items": args.items,
            "question": args.question,
            "results": results,
        }, ensure_ascii=False, indent=2, default=str))
    else:
        print_results(results, args.items, args.question)


if __name__ == "__main__":
    main()
