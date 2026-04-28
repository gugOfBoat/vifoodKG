"""
ViFoodKG offline statistics report from a Neo4j triple export.

This script reads data/triples/extracted_triples_neo4j.json, which is produced
by src/utils/extract_triples_neo4j.py, and reports graph statistics without
connecting to Neo4j.

Usage:
  python src/utils/check_kg_export.py
  python src/utils/check_kg_export.py --json
  python src/utils/check_kg_export.py --input data/triples/extracted_triples_neo4j.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = KG_ROOT / "data" / "triples" / "extracted_triples_neo4j.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print offline KG statistics from extracted_triples_neo4j.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the Neo4j triple export JSON file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of a human-readable table.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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

    raise ValueError("Expected a JSON array of triple objects.")


def source_bucket(source_url: str) -> str:
    if not source_url:
        return "Missing source"
    if source_url.startswith("http"):
        return "Web source"
    if source_url in {"Cognitive_Reasoning", "Common_Sense", "LLM_Knowledge"}:
        return "LLM reasoning"
    return "Other"


def sorted_counter(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"name": name, "count": count}
        for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def collect_stats(triples: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: set[tuple[str, str]] = set()
    edges: set[tuple[str, str, str, str, str]] = set()
    invalid_rows = 0

    nodes_by_type: dict[str, set[str]] = defaultdict(set)
    relation_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    subject_type_counter: Counter[str] = Counter()
    target_type_counter: Counter[str] = Counter()

    evidence_count = 0
    verbalized_text_count = 0

    for triple in triples:
        subject = clean_text(triple.get("subject"))
        subject_type = clean_text(triple.get("subject_type")) or "Unknown"
        relation = clean_text(triple.get("relation"))
        target = clean_text(triple.get("target"))
        target_type = clean_text(triple.get("target_type")) or "Unknown"
        source_url = clean_text(triple.get("source_url"))

        if not subject or not relation or not target:
            invalid_rows += 1
            continue

        subject_node = (subject_type, subject)
        target_node = (target_type, target)
        nodes.add(subject_node)
        nodes.add(target_node)
        nodes_by_type[subject_type].add(subject)
        nodes_by_type[target_type].add(target)

        edge = (subject_type, subject, relation, target_type, target)
        edges.add(edge)

        relation_counter[relation] += 1
        source_counter[source_bucket(source_url)] += 1
        subject_type_counter[subject_type] += 1
        target_type_counter[target_type] += 1

        if clean_text(triple.get("evidence")):
            evidence_count += 1
        if clean_text(triple.get("verbalized_text")):
            verbalized_text_count += 1

    entity_types = Counter({label: len(names) for label, names in nodes_by_type.items()})

    return {
        "input_rows": len(triples),
        "valid_edges": sum(relation_counter.values()),
        "unique_edges": len(edges),
        "duplicate_edges": sum(relation_counter.values()) - len(edges),
        "connected_nodes": len(nodes),
        "entity_type_count": len(entity_types),
        "relationship_type_count": len(relation_counter),
        "invalid_rows": invalid_rows,
        "evidence_edges": evidence_count,
        "verbalized_text_edges": verbalized_text_count,
        "entity_types": sorted_counter(entity_types),
        "relationship_types": sorted_counter(relation_counter),
        "source_distribution": sorted_counter(source_counter),
        "subject_type_distribution": sorted_counter(subject_type_counter),
        "target_type_distribution": sorted_counter(target_type_counter),
        "note": "This export contains only relationships, so isolated Neo4j nodes cannot be counted offline.",
    }


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    if not rows:
        print("  (empty)")
        return

    name_width = max(len(str(row["name"])) for row in rows)
    for row in rows:
        print(f"  - {row['name']:<{name_width}} : {row['count']}")


def print_report(stats: dict[str, Any], input_path: Path) -> None:
    print("=" * 72)
    print("ViFoodKG Offline Knowledge Graph Statistics")
    print("=" * 72)
    print(f"Input file             : {input_path}")
    print(f"Input rows             : {stats['input_rows']}")
    print(f"Valid edges            : {stats['valid_edges']}")
    print(f"Unique edges           : {stats['unique_edges']}")
    print(f"Duplicate edges        : {stats['duplicate_edges']}")
    print(f"Connected nodes        : {stats['connected_nodes']}")
    print(f"Entity types           : {stats['entity_type_count']}")
    print(f"Relationship types     : {stats['relationship_type_count']}")
    print(f"Invalid rows           : {stats['invalid_rows']}")
    print(f"Edges with evidence    : {stats['evidence_edges']}")
    print(f"Edges with text        : {stats['verbalized_text_edges']}")
    print("Isolated nodes         : not available from edge-only export")
    print()

    print_table("Entity types by node label:", stats["entity_types"])
    print()
    print_table("Relationship types by edge type:", stats["relationship_types"])
    print()
    print_table("Source distribution:", stats["source_distribution"])
    print()
    print_table("Subject type distribution:", stats["subject_type_distribution"])
    print()
    print_table("Target type distribution:", stats["target_type_distribution"])
    print()
    print(f"Note: {stats['note']}")
    print("=" * 72)


def main() -> int:
    args = parse_args()

    try:
        triples = load_export(args.input)
        stats = collect_stats(triples)
    except Exception as exc:
        print(f"[ERROR] Failed to collect offline KG stats: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print_report(stats, args.input)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
