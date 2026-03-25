"""
ViFoodKG — KGRetriever Module
===============================
Chiến lược query: Neo → Traverse → Prefilter → Rank

  1. Neo       : Anchor vào đúng các Dish node trong items
  2. Traverse  : Lấy neighborhood 1-hop / 2-hop quanh các Dish đó
  3. Prefilter : Giữ lại đúng relation phục vụ qtype hiện tại
  4. Rank      : So khớp vector giữa query intent và full path text

Khác biệt chính so với bản cũ:
- Không rank edge embedding đơn lẻ nữa; rank theo full path text để 2-hop không bị lép vế
- Có thể truyền allowed_relations để lọc theo qtype trước khi top-k
- Query text nên chỉ chứa intent của qtype; tên món đã được dùng ở bước anchor/traverse
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterable, Optional

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
  AND NOT type(r) IN ['hasSubRule']

RETURN DISTINCT
  d.name            AS subject,
  'Dish'            AS subject_type,
  type(r)           AS relation,
  target.name       AS target,
  labels(target)[0] AS target_type,
  null              AS via,
  null              AS via_type,
  r.verbalized_text AS verbalized_text,
  r.evidence        AS evidence,
  r.source_url      AS source_url,
  1                 AS hop

UNION ALL

// ── 2-hop: Dish → Ingredient → target ──────────────────────────────────────
MATCH (d:Dish)-[:hasIngredient]->(ing:Ingredient)-[r]->(target)
WHERE d.name IN $items

RETURN DISTINCT
  d.name            AS subject,
  'Dish'            AS subject_type,
  type(r)           AS relation,
  target.name       AS target,
  labels(target)[0] AS target_type,
  ing.name          AS via,
  'Ingredient'      AS via_type,
  r.verbalized_text AS verbalized_text,
  r.evidence        AS evidence,
  r.source_url      AS source_url,
  2                 AS hop

UNION ALL

// ── SubstitutionRule: Dish → SubRule → Ingredient ──────────────────────────
MATCH (d:Dish)-[:hasSubRule]->(sr:SubstitutionRule)-[r]->(ing:Ingredient)
WHERE d.name IN $items

RETURN DISTINCT
  d.name            AS subject,
  'Dish'            AS subject_type,
  type(r)           AS relation,
  ing.name          AS target,
  'Ingredient'      AS target_type,
  sr.name           AS via,
  'SubstitutionRule' AS via_type,
  r.verbalized_text AS verbalized_text,
  r.evidence        AS evidence,
  r.source_url      AS source_url,
  2                 AS hop
"""


_RELATION_TO_VI = {
    "hasIngredient": "có thành phần",
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


def _relation_to_vi(relation: str) -> str:
    return _RELATION_TO_VI.get((relation or "").strip(), (relation or "").strip())


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _build_rank_text(row: dict) -> str:
    subject = _norm_text(row.get("subject"))
    relation = _norm_text(row.get("relation"))
    relation_vi = _relation_to_vi(relation)
    target = _norm_text(row.get("target"))
    via = _norm_text(row.get("via"))

    if int(row.get("hop") or 0) == 1:
        return f"{subject} {relation_vi} {target}".strip()

    if relation in {"fromIngredient", "toIngredient"}:
        return f"{subject} có quy tắc thay thế {via}; {via} {relation_vi} {target}".strip()

    return f"{subject} có thành phần {via}; {via} {relation_vi} {target}".strip()


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
        self._text_embedding_cache: Dict[str, np.ndarray] = {}
        print("[KGRetriever] Ready.\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [E5_QUERY_PREFIX + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0]

    def _embed_many(self, texts: list[str]) -> list[np.ndarray]:
        uncached = [text for text in texts if text not in self._text_embedding_cache]
        if uncached:
            vectors = self._model.encode(
                [E5_QUERY_PREFIX + text for text in uncached],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vec in zip(uncached, vectors, strict=True):
                self._text_embedding_cache[text] = vec
        return [self._text_embedding_cache[text] for text in texts]

    def _cosine(self, vec_a: np.ndarray, vec_b: np.ndarray | list) -> float:
        b = np.array(vec_b, dtype=np.float32)
        return float(np.dot(vec_a, b))

    # ── Main API ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        items: list[str],
        question: str,
        top_k: int = 5,
        allowed_relations: Optional[Iterable[str]] = None,
    ) -> list[dict]:
        """
        Parameters
        ----------
        items             : dish names used to anchor the local subgraph
        question          : qtype intent / detail text (should be short, relation-focused)
        top_k             : number of rows to return after ranking
        allowed_relations : optional relation whitelist applied BEFORE top-k

        Returns
        -------
        List[dict] sorted by score desc.
        Each row additionally includes `rank_text`, the full path text used for scoring.
        """
        query_vec = self._embed(question)

        with self._driver.session() as session:
            raw = session.run(_TRAVERSE_QUERY, items=items)
            rows = [dict(r) for r in raw]

        if not rows:
            return []

        if allowed_relations:
            allowed = {_norm_text(rel) for rel in allowed_relations if _norm_text(rel)}
            rows = [row for row in rows if _norm_text(row.get("relation")) in allowed]
            if not rows:
                return []

        rank_texts = [_build_rank_text(row) for row in rows]
        rank_vecs = self._embed_many(rank_texts)

        scored = []
        for row, rank_text, rank_vec in zip(rows, rank_texts, rank_vecs, strict=True):
            scored.append({
                **row,
                "rank_text": rank_text,
                "score": round(self._cosine(query_vec, rank_vec), 6),
            })

        scored.sort(key=lambda x: (x["score"], x.get("hop", 0)), reverse=True)
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

        rt = r.get("rank_text")
        if rt:
            print(f"    ≈ {rt}")
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
    parser.add_argument("--relations", nargs="*", default=[], help="Optional relation whitelist before ranking")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with KGRetriever(device=args.device) as kg:
        results = kg.retrieve(args.items, args.question, args.top_k, allowed_relations=args.relations)

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
