"""
ViFoodKG — Check Knowledge Graph Status 
=======================================
Run this script to retrieve current Neo4j statistics and verify the
graph's density and distribution after ingestion or enrichment phases.

Usage:
  python src/check_kg.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

def main():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[ERROR] neo4j driver not installed. Run: pip install neo4j")
        sys.exit(1)

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pw = os.getenv("NEO4J_PASSWORD")

    if not uri or not pw:
        print("[ERROR] Missing NEO4J_URI or NEO4J_PASSWORD in .env")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, pw))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[ERROR] Failed to connect to Neo4j: {e}")
        sys.exit(1)

    print("=========================================================")
    print("      VIFOODKG - KNOWLEDGE GRAPH STATUS REPORT")
    print("=========================================================\n")

    with driver.session() as s:
        # Tổng quan
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        dishes = s.run("MATCH (d:Dish) RETURN count(d) AS c").single()["c"]
        print(f"[*] TOTAL GRAPH SIZE")
        print(f"    - Nodes: {nodes}")
        print(f"    - Relationships: {rels}")
        print(f"    - Dish Nodes: {dishes}\n")

        # Độ phủ (Sparsity)
        r = s.run(
            "MATCH (d:Dish)-[r]-() "
            "WITH d, count(r) AS c "
            "RETURN min(c) AS mn, round(avg(c),2) AS av, max(c) AS mx, count(d) AS n"
        ).single()
        print(f"[*] DENSITY (Relationships per Dish)")
        print(f"    - Average: {r['av']} rels/dish")
        print(f"    - Min: {r['mn']} | Max: {r['mx']}")
        
        orphans = s.run("MATCH (d:Dish) WHERE NOT (d)-[]-() RETURN count(d) AS c").single()["c"]
        print(f"    - Orphan Dishes (0 relations): {orphans}\n")

        # Phân bố nguồn dữ liệu (Source Distribution)
        print(f"[*] DATA SOURCING")
        q_src = (
            "MATCH ()-[r]->() WHERE r.source_url IS NOT NULL "
            "WITH CASE "
            "  WHEN r.source_url STARTS WITH 'http' THEN 'Web (Wikipedia/Blogs)' "
            "  WHEN r.source_url IN ['Cognitive_Reasoning', 'Common_Sense', 'LLM_Knowledge'] THEN 'LLM Expert Reasoning' "
            "  ELSE 'Other' END AS src, r "
            "RETURN src, count(r) AS c ORDER BY c DESC"
        )
        for row in s.run(q_src):
             print(f"    - {row['src']:25s}: {row['c']} triples")
        print()

        # Phân bố quan hệ (Relation Target Gaps)
        print(f"[*] ONTOLOGY COVERAGE (Relationships by Type)")
        q_rel = "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC"
        for row in s.run(q_rel):
            print(f"    - {row['t']:20s}: {row['c']}")
        print()

    driver.close()
    print("=========================================================")

if __name__ == "__main__":
    main()
