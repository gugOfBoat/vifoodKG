"""
ViFoodKG — Multi-Entity Knowledge Retriever
=============================================
Truy xuất triples phù hợp nhất từ Neo4j bằng kỹ thuật lai:
  Graph Filter (lọc theo items) + Vector Search (cosine trên embedding).

3 tham số bắt buộc:
  --items        : Danh sách thực thể nhận diện từ ảnh
  --question     : Mô tả chi tiết loại câu hỏi (free-text)
  --top-k        : Số lượng triples trả về (default: 5)

Pipeline:
  1. Gộp items + question_desc → "query: {question}. Các đối tượng: {items}"
  2. Nhúng chuỗi trên bằng intfloat/multilingual-e5-small
  3. Cypher:  lọc edges có ít nhất 1 đầu ∈ items
            → vector similarity trên edge.embedding
            → ưu tiên edges có cả 2 đầu ∈ items (bonus score)
  4. Trả về Top-K: subject, relation, target, verbalized_text, evidence

Usage (Colab / Local):
  python src/query.py -i "Phở Bò" "Thịt Bò" "Rau Thơm" -q "nguyên liệu chính" -k 5"""

import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

MODEL_NAME = "intfloat/multilingual-e5-small"

# ══════════════════════════════════════════════════════════════════════════════
# 1. EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

_model = None   # lazy singleton


def get_model(device: str = "auto"):
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    _model = SentenceTransformer(MODEL_NAME, device=device)
    print("  Model loaded.\n")
    return _model


def embed_query(items: list[str], question_desc: str, device: str = "auto") -> list[float]:
    """Tạo vector cho câu hỏi, kết hợp context từ items."""
    model = get_model(device)
    items_str = ", ".join(items)
    # E5 query prefix — khác với "passage:" dùng khi index
    text = f"query: {question_desc}. Các đối tượng: {items_str}"
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 2. HYBRID CYPHER — graph filter + vector search + internal-link bonus
# ══════════════════════════════════════════════════════════════════════════════

HYBRID_QUERY = """
// Step 1: Vector search trên toàn bộ edges đã embed
CALL db.index.vector.queryRelationships(
  'triple_vector_index', $top_k_fetch, $query_vec
) YIELD relationship AS r, score

// Step 2: Lấy subject và target
MATCH (a)-[r]->(b)

// Step 3: Graph filter — ít nhất 1 đầu nằm trong items
WHERE a.name IN $items OR b.name IN $items

// Step 4: Tính điểm ưu tiên nội bộ
WITH a, r, b, score,
     CASE WHEN a.name IN $items AND b.name IN $items THEN 0.15 ELSE 0.0 END AS bonus

RETURN a.name          AS subject,
       labels(a)[0]    AS subject_type,
       type(r)         AS relation,
       b.name          AS target,
       labels(b)[0]    AS target_type,
       r.verbalized_text AS verbalized_text,
       r.evidence       AS evidence,
       r.source_url     AS source_url,
       score + bonus    AS final_score
ORDER BY final_score DESC
LIMIT $top_k
"""


def search(driver, query_vec: list[float], items: list[str], top_k: int) -> list[dict]:
    """Run hybrid vector + graph query."""
    # Fetch more than top_k to account for graph filter dropping some rows
    top_k_fetch = max(top_k * 5, 50)

    with driver.session() as session:
        result = session.run(
            HYBRID_QUERY,
            query_vec=query_vec,
            items=items,
            top_k=top_k,
            top_k_fetch=top_k_fetch,
        )
        return [dict(r) for r in result]


# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_results(records: list[dict], items: list[str], question: str) -> None:
    print(f"\n{'═'*70}")
    print(f"  Items    : {', '.join(items)}")
    print(f"  Question : {question}")
    print(f"  Results  : {len(records)} triples")
    print(f"{'═'*70}\n")

    if not records:
        print("  (Không tìm thấy kết quả phù hợp)\n")
        return

    for i, row in enumerate(records, 1):
        score = row.get("final_score", 0)
        subj  = row.get("subject", "?")
        rel   = row.get("relation", "?")
        tgt   = row.get("target", "?")
        st    = row.get("subject_type", "")
        tt    = row.get("target_type", "")

        print(f"  #{i}  (score: {score:.4f})")
        print(f"    ({st}) {subj}  ──[{rel}]──▶  {tgt} ({tt})")

        vt = row.get("verbalized_text")
        if vt:
            print(f"    ✎ {vt}")

        ev = row.get("evidence")
        if ev:
            print(f"    📖 {ev[:150]}{'...' if len(ev) > 150 else ''}")

        src = row.get("source_url")
        if src and src != "LLM_Knowledge":
            print(f"    🔗 {src}")

        print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. PROGRAMMATIC API — for import from other scripts / Colab
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_triples(
    items: list[str],
    question_desc: str,
    top_k: int = 5,
    device: str = "auto",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> list[dict]:
    """
    Public API — call from any Python script or notebook.

    Returns list of dicts with keys:
      subject, subject_type, relation, target, target_type,
      verbalized_text, evidence, source_url, final_score
    """
    from neo4j import GraphDatabase

    uri  = neo4j_uri  or NEO4J_URI
    user = neo4j_user or NEO4J_USER
    pw   = neo4j_password or NEO4J_PASSWORD

    query_vec = embed_query(items, question_desc, device=device)
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    try:
        driver.verify_connectivity()
        return search(driver, query_vec, items, top_k)
    finally:
        driver.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ViFoodKG — Multi-Entity Knowledge Retriever",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/query.py -i "Phở Bò" "Thịt Bò" -q "nguyên liệu chính" -k 5
  python src/query.py -i "Bánh Xèo" "Tôm" -q "dị ứng" -k 3 --json
  python src/query.py -i "Bún Chả" "Chả Lụa" -q "ăn kèm với gì" -k 10
""",
    )
    parser.add_argument("--items", "-i", nargs="+", required=True,
                        metavar="ITEM", help="Danh sách thực thể (món ăn, nguyên liệu)")
    parser.add_argument("--question", "-q", required=True,
                        metavar="DESC", help="Mô tả câu hỏi / ý định truy vấn")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Số triples trả về (default: 5)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device cho embedding model")
    parser.add_argument("--json", action="store_true",
                        help="Xuất JSON thay vì bảng")
    args = parser.parse_args()

    if not NEO4J_PASSWORD:
        print("[ERROR] NEO4J_PASSWORD not set in .env")
        sys.exit(1)

    # Embed
    query_vec = embed_query(args.items, args.question, device=args.device)

    # Connect & search
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        records = search(driver, query_vec, args.items, args.top_k)
    finally:
        driver.close()

    # Output
    if args.json:
        out = {
            "items": args.items,
            "question": args.question,
            "top_k": args.top_k,
            "results": records,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        print_results(records, args.items, args.question)


if __name__ == "__main__":
    main()
