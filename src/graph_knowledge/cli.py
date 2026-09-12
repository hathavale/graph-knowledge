"""Ingest text and inspect the resulting graph."""

from __future__ import annotations

import argparse
import sys

from graph_knowledge.extraction import RuleBasedExtractor
from graph_knowledge.pipeline import Pipeline
from graph_knowledge.store.base import GraphStore


def build_store(backend: str, path: str) -> GraphStore:
    if backend == "neo4j":
        from graph_knowledge.store.neo4j_store import Neo4jStore

        return Neo4jStore()
    from graph_knowledge.store.embedded_store import EmbeddedStore

    return EmbeddedStore(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graph-knowledge")
    parser.add_argument("text", nargs="?", help="text to ingest; omit to read stdin")
    parser.add_argument("--doc-id", default="cli", help="document id (scopes unnamed entities)")
    parser.add_argument("--backend", choices=["neo4j", "embedded"], default="neo4j")
    parser.add_argument("--path", default="./data/graph", help="embedded backend path")
    parser.add_argument("--person", help="after ingesting, show this person's graph")
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("no text supplied")

    store = build_store(args.backend, args.path)
    store.initialize()
    try:
        extraction = Pipeline(RuleBasedExtractor(), store).ingest(text, args.doc_id)

        print(f"Extracted from {args.doc_id!r}:")
        for person in extraction.people:
            print(f"  Person   name={person.name!r} gender={person.gender!r}")
        for travel in extraction.travels:
            print(f"  Travel   {travel.person.name} -> {travel.place.type or travel.place.name}")
        for event in extraction.activities:
            place = event.place.type or event.place.name if event.place else None
            attrs = ", ".join(
                f"{a.name}={a.value!r}" for a in event.activity.attributes
            )
            print(f"  Activity {event.person.name} :: {event.activity.type!r} at={place!r} [{attrs}]")

        name = args.person or (extraction.people[0].name if extraction.people else None)
        if name:
            print(f"\nGraph for {name!r}:")
            for row in store.travels_for(name):
                print(f"  TRAVEL   -> Place(type={row.place_type!r}) at {row.travel_datetime}")
            for row in store.activities_for(name):
                print(
                    f"  ACTIVITY {row.activity_type!r} at Place(type={row.place_type!r})"
                    f" attribute {row.attribute_name!r}={row.attribute_value!r}"
                )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
