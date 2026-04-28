"""
Restore a Neo4j Aura knowledge graph from a flat relationship export.

This script imports the JSON produced by:
  src/utils/extract_triples_neo4j.py

Expected input shape:
[
  {
    "subject": "Banh Beo",
    "subject_type": "Dish",
    "relation": "hasIngredient",
    "target": "Bot Gao",
    "target_type": "Ingredient",
    "evidence": "...",
    "source_url": "...",
    "verbalized_text": "..."
  }
]

Usage:
  python src/utils/restore_neo4j_export.py --dry-run
  python src/utils/restore_neo4j_export.py
  python src/utils/restore_neo4j_export.py --input data/triples/extracted_triples_neo4j.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


KG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = KG_ROOT / ".env"
DEFAULT_INPUT = KG_ROOT / "data" / "triples" / "extracted_triples_neo4j.json"
DEFAULT_BATCH_SIZE = 500
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore ViFoodKG Neo4j graph from a flat JSON relationship export."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to flat Neo4j export JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f"Path to .env containing NEO4J_URI/USERNAME/PASSWORD (default: {DEFAULT_ENV_FILE})",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Optional Neo4j database name. AuraDB usually uses the default database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per write transaction (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Import only the first N valid relationships. Useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print an import plan without connecting to Neo4j.",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Do not create name indexes before import.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def nullable_text(value: Any) -> str | None:
    text = clean_text(value)
    return text or None


def safe_identifier(raw: str, kind: str) -> str:
    value = clean_text(raw)
    if not value:
        raise ValueError(f"Missing {kind}")
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe {kind} for Cypher identifier: {value!r}")
    return value


def load_export(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        rows = data.get("triples") or data.get("rows") or data.get("data")
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]

    raise ValueError("Expected a JSON array of relationship objects.")


def normalize_rows(
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    rows: list[dict[str, Any]] = []
    invalid: list[tuple[int, str]] = []
    dedup: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    for idx, item in enumerate(raw_rows, 1):
        subject = clean_text(item.get("subject"))
        relation = clean_text(item.get("relation"))
        target = clean_text(item.get("target"))

        if not subject or not relation or not target:
            invalid.append((idx, "missing subject/relation/target"))
            continue

        try:
            subject_type = safe_identifier(
                item.get("subject_type") or "Entity",
                "subject_type",
            )
            rel_type = safe_identifier(relation, "relation")
            target_type = safe_identifier(
                item.get("target_type") or "Entity",
                "target_type",
            )
        except ValueError as exc:
            invalid.append((idx, str(exc)))
            continue

        row = {
            "subject": subject,
            "subject_type": subject_type,
            "relation": rel_type,
            "target": target,
            "target_type": target_type,
            "evidence": nullable_text(item.get("evidence")),
            "source_url": nullable_text(item.get("source_url")),
            "verbalized_text": nullable_text(item.get("verbalized_text")),
        }

        key = (subject_type, subject, rel_type, target_type, target)
        if key not in dedup:
            dedup[key] = row
            continue

        existing = dedup[key]
        for prop in ("evidence", "source_url", "verbalized_text"):
            if not existing.get(prop) and row.get(prop):
                existing[prop] = row[prop]

    rows = list(dedup.values())
    return rows, invalid


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    if limit < 1:
        raise ValueError("--limit must be greater than 0 when provided")
    return rows[:limit]


def group_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["subject_type"], row["relation"], row["target_type"])
        groups[key].append(row)
    return groups


def chunks(seq: list[dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def print_plan(
    input_path: Path,
    raw_count: int,
    rows: list[dict[str, Any]],
    invalid: list[tuple[int, str]],
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> None:
    labels = Counter()
    relations = Counter()
    for row in rows:
        labels[row["subject_type"]] += 1
        labels[row["target_type"]] += 1
        relations[row["relation"]] += 1

    print("=" * 72)
    print("ViFoodKG Neo4j Restore Plan")
    print("=" * 72)
    print(f"Input file        : {input_path}")
    print(f"Input rows        : {raw_count}")
    print(f"Valid rows        : {len(rows)}")
    print(f"Invalid rows      : {len(invalid)}")
    print(f"Write groups      : {len(groups)}")
    print(f"Node labels       : {len(labels)}")
    print(f"Relationship types: {len(relations)}")
    print()

    print("Relationship counts:")
    for rel, count in relations.most_common():
        print(f"  - {rel:<20} {count}")
    print()

    if invalid:
        print("Invalid row samples:")
        for line_no, reason in invalid[:10]:
            print(f"  - row {line_no}: {reason}")
        if len(invalid) > 10:
            print(f"  ... {len(invalid) - 10} more invalid rows")
        print()

    print("=" * 72)


def load_env(env_file: Path) -> None:
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()


def get_neo4j_config() -> tuple[str, str, str]:
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
        raise RuntimeError(f"Missing required Neo4j env var(s): {', '.join(missing)}")

    return uri, username, password


def make_driver():
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "neo4j driver not installed. Run: python -m pip install neo4j"
        ) from exc

    uri, username, password = get_neo4j_config()
    return GraphDatabase.driver(uri, auth=(username, password))


def session_kwargs(database: str | None) -> dict[str, str]:
    if database:
        return {"database": database}
    return {}


def create_name_indexes(session: Any, labels: set[str]) -> None:
    for label in sorted(labels):
        safe_label = safe_identifier(label, "label")
        session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{safe_label}) ON (n.name)")


def write_relationship_batch(
    tx: Any,
    subject_type: str,
    relation: str,
    target_type: str,
    rows: list[dict[str, Any]],
) -> None:
    subject_label = safe_identifier(subject_type, "subject_type")
    rel_type = safe_identifier(relation, "relation")
    target_label = safe_identifier(target_type, "target_type")

    query = f"""
    UNWIND $rows AS row
    MERGE (s:{subject_label} {{name: row.subject}})
    MERGE (t:{target_label} {{name: row.target}})
    MERGE (s)-[r:{rel_type}]->(t)
    SET r.evidence = row.evidence,
        r.source_url = row.source_url,
        r.verbalized_text = row.verbalized_text
    """
    tx.run(query, rows=rows)


def restore_graph(
    driver: Any,
    database: str | None,
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    batch_size: int,
    skip_indexes: bool,
) -> None:
    if batch_size < 1:
        raise ValueError("--batch-size must be greater than 0")

    with driver.session(**session_kwargs(database)) as session:
        if not skip_indexes:
            labels = {label for s, _, t in groups for label in (s, t)}
            print(f"Creating name indexes for {len(labels)} labels...")
            create_name_indexes(session, labels)
            print("Indexes ready.")

        total = sum(len(items) for items in groups.values())
        done = 0
        for subject_type, relation, target_type in sorted(groups):
            items = groups[(subject_type, relation, target_type)]
            for batch in chunks(items, batch_size):
                session.execute_write(
                    write_relationship_batch,
                    subject_type,
                    relation,
                    target_type,
                    batch,
                )
                done += len(batch)
                print(f"Imported {done}/{total} relationships", end="\r")

        print(f"Imported {done}/{total} relationships")


def collect_db_counts(driver: Any, database: str | None) -> dict[str, Any]:
    with driver.session(**session_kwargs(database)) as session:
        record = session.run(
            """
            MATCH (n)
            WITH count(n) AS nodes
            MATCH ()-[r]->()
            RETURN nodes, count(r) AS relationships
            """
        ).single()
        rels = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC, type
            """
        )
        return {
            "nodes": int(record["nodes"]) if record else 0,
            "relationships": int(record["relationships"]) if record else 0,
            "relationship_types": [
                {"type": row["type"], "count": row["count"]} for row in rels
            ],
        }


def main() -> int:
    args = parse_args()

    try:
        raw_rows = load_export(args.input)
        rows, invalid = normalize_rows(raw_rows)
        rows = limit_rows(rows, args.limit)
        groups = group_rows(rows)
        print_plan(args.input, len(raw_rows), rows, invalid, groups)

        if invalid:
            print("[ERROR] Invalid rows found. Fix the export before importing.")
            return 1

        if args.dry_run:
            print("Dry-run only. No Neo4j connection was opened.")
            return 0

        load_env(args.env)
        driver = make_driver()
        try:
            print("Verifying Neo4j connectivity...")
            driver.verify_connectivity()
            print("Connection OK.")

            restore_graph(
                driver=driver,
                database=args.database,
                groups=groups,
                batch_size=args.batch_size,
                skip_indexes=args.skip_indexes,
            )

            counts = collect_db_counts(driver, args.database)
            print()
            print("Restore complete.")
            print(f"Neo4j nodes        : {counts['nodes']}")
            print(f"Neo4j relationships: {counts['relationships']}")
            print("Relationship types:")
            for row in counts["relationship_types"]:
                print(f"  - {row['type']:<20} {row['count']}")
        finally:
            driver.close()

    except Exception as exc:
        print(f"[ERROR] Restore failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
