"""
ViFoodKG Neo4j statistics report.

Counts the current Neo4j knowledge graph size:
- total nodes
- total edges / relationships
- entity types (Neo4j node labels)
- relationship types
- dish coverage and source distribution

Usage:
  python src/utils/check_kg.py
  python src/utils/check_kg.py --json
  python src/utils/check_kg.py --env .env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


KG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = KG_ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print basic Neo4j KG statistics for ViFoodKG."
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to .env file that contains NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of a human-readable table.",
    )
    return parser.parse_args()


def load_env(env_file: Path) -> None:
    if env_file.exists():
        load_dotenv(env_file)
        return

    load_dotenv()


def require_neo4j_config() -> tuple[str, str, str]:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    missing = [
        name
        for name, value in {
            "NEO4J_URI": uri,
            "NEO4J_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing required Neo4j env var(s): {missing_text}")

    return uri, username, password


def run_scalar(session: Any, query: str) -> int:
    record = session.run(query).single()
    if record is None:
        return 0
    return int(record["count"])


def collect_stats(driver: Any) -> dict[str, Any]:
    with driver.session() as session:
        total_nodes = run_scalar(session, "MATCH (n) RETURN count(n) AS count")
        total_edges = run_scalar(session, "MATCH ()-[r]->() RETURN count(r) AS count")
        dish_nodes = run_scalar(session, "MATCH (d:Dish) RETURN count(d) AS count")
        isolated_nodes = run_scalar(
            session,
            "MATCH (n) WHERE NOT (n)--() RETURN count(n) AS count",
        )

        dish_density_record = session.run(
            """
            MATCH (d:Dish)
            OPTIONAL MATCH (d)-[r]-()
            WITH d, count(r) AS degree
            RETURN
              min(degree) AS min_degree,
              round(avg(degree), 2) AS avg_degree,
              max(degree) AS max_degree,
              sum(CASE WHEN degree = 0 THEN 1 ELSE 0 END) AS orphan_dishes
            """
        ).single()
        dish_density = dish_density_record.data() if dish_density_record else {}

        labels = [
            {"label": row["label"], "count": row["count"]}
            for row in session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(n) AS count
                ORDER BY count DESC, label
                """
            )
        ]

        label_sets = [
            {"labels": row["labels"], "count": row["count"]}
            for row in session.run(
                """
                MATCH (n)
                RETURN labels(n) AS labels, count(n) AS count
                ORDER BY count DESC
                """
            )
        ]

        relationship_types = [
            {"type": row["type"], "count": row["count"]}
            for row in session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC, type
                """
            )
        ]

        source_distribution = [
            {"source": row["source"], "count": row["count"]}
            for row in session.run(
                """
                MATCH ()-[r]->()
                WITH CASE
                  WHEN r.source_url IS NULL OR trim(toString(r.source_url)) = ''
                    THEN 'Missing source'
                  WHEN toString(r.source_url) STARTS WITH 'http'
                    THEN 'Web source'
                  WHEN r.source_url IN [
                    'Cognitive_Reasoning',
                    'Common_Sense',
                    'LLM_Knowledge'
                  ]
                    THEN 'LLM reasoning'
                  ELSE 'Other'
                END AS source
                RETURN source, count(*) AS count
                ORDER BY count DESC, source
                """
            )
        ]

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "dish_nodes": dish_nodes,
        "entity_type_count": len(labels),
        "relationship_type_count": len(relationship_types),
        "isolated_nodes": isolated_nodes,
        "dish_density": {
            "min_relationships_per_dish": dish_density.get("min_degree", 0),
            "avg_relationships_per_dish": dish_density.get("avg_degree", 0),
            "max_relationships_per_dish": dish_density.get("max_degree", 0),
            "orphan_dishes": dish_density.get("orphan_dishes", 0),
        },
        "entity_types": labels,
        "node_label_sets": label_sets,
        "relationship_types": relationship_types,
        "source_distribution": source_distribution,
    }


def print_table(title: str, rows: list[dict[str, Any]], key_name: str) -> None:
    print(title)
    if not rows:
        print("  (empty)")
        return

    name_width = max(len(str(row[key_name])) for row in rows)
    for row in rows:
        name = str(row[key_name])
        print(f"  - {name:<{name_width}} : {row['count']}")


def print_report(stats: dict[str, Any]) -> None:
    print("=" * 64)
    print("ViFoodKG Neo4j Knowledge Graph Statistics")
    print("=" * 64)
    print(f"Total nodes          : {stats['total_nodes']}")
    print(f"Total edges          : {stats['total_edges']}")
    print(f"Dish nodes           : {stats['dish_nodes']}")
    print(f"Entity types         : {stats['entity_type_count']}")
    print(f"Relationship types   : {stats['relationship_type_count']}")
    print(f"Isolated nodes       : {stats['isolated_nodes']}")
    print(
        "Dish rels/dish      : "
        f"avg={stats['dish_density']['avg_relationships_per_dish']}, "
        f"min={stats['dish_density']['min_relationships_per_dish']}, "
        f"max={stats['dish_density']['max_relationships_per_dish']}, "
        f"orphan={stats['dish_density']['orphan_dishes']}"
    )
    print()

    print_table("Entity types by node label:", stats["entity_types"], "label")
    print()
    print_table("Relationship types by edge type:", stats["relationship_types"], "type")
    print()
    print_table("Source distribution:", stats["source_distribution"], "source")
    print()

    if stats["node_label_sets"]:
        print("Node label sets:")
        for row in stats["node_label_sets"]:
            labels = ":".join(row["labels"]) if row["labels"] else "(no label)"
            print(f"  - {labels} : {row['count']}")
        print()

    print("=" * 64)


def main() -> int:
    args = parse_args()
    load_env(args.env)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[ERROR] neo4j driver not installed. Run: pip install neo4j", file=sys.stderr)
        return 1

    try:
        uri, username, password = require_neo4j_config()
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        stats = collect_stats(driver)
    except Exception as exc:
        print(f"[ERROR] Failed to collect Neo4j stats: {exc}", file=sys.stderr)
        return 1
    finally:
        if "driver" in locals():
            driver.close()

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print_report(stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
