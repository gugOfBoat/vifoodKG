import os
import json
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "triples" / "extracted_triples_neo4j.json"

QUERY = """
MATCH (s)-[r]->(t)
RETURN
  s.name AS subject,
  labels(s)[0] AS subject_type,
  type(r) AS relation,
  t.name AS target,
  labels(t)[0] AS target_type,
  r.evidence AS evidence,
  r.source_url AS source_url,
  r.verbalized_text AS verbalized_text
ORDER BY subject, relation, target
"""

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with driver.session() as session:
    rows = session.run(QUERY).data()

driver.close()

with open(DEFAULT_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"Exported {len(rows)} triples to {DEFAULT_OUTPUT}")