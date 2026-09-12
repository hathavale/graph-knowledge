"""Ingest a ticket and show what the graph learned."""

from __future__ import annotations

import argparse
import sys

from graph_knowledge.extraction import LLMExtractor
from graph_knowledge.extraction.llm import DEFAULT_MODEL
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
    parser.add_argument("text", nargs="?", help="ticket text; omit to read stdin")
    parser.add_argument("--ticket-id", default="CLI-1")
    parser.add_argument("--backend", choices=["neo4j", "embedded"], default="neo4j")
    parser.add_argument("--path", default="./data/graph", help="embedded backend path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--topic", help="after ingesting, show candidates for this topic")
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("no text supplied")

    store = build_store(args.backend, args.path)
    store.initialize()
    try:
        vocabulary = store.load_vocabulary()
        pipeline = Pipeline(
            LLMExtractor(vocabulary=vocabulary, model=args.model, effort=args.effort),
            store,
            vocabulary,
        )
        extraction = pipeline.ingest(text, args.ticket_id)

        print(f"Ticket {extraction.ticket.id}:")
        for person in extraction.people:
            aliases = f" ({', '.join(person.aliases)})" if person.aliases else ""
            print(f"  Person       {person.name}{aliases}")
        for topic in extraction.topics:
            print(f"  Topic        {topic.name}")
        for p in extraction.participations:
            print(f"  Participated {p.person.name} as {p.role.value}")
        for m in extraction.mentions:
            about = f" about {m.topic.name!r}" if m.topic else ""
            confidence = m.evidence.confidence if m.evidence else None
            print(f"  Mentioned    {m.person.name} as {m.role.value}{about} ({confidence})")
            if m.evidence:
                print(f"               “{m.evidence.excerpt}”")
        for f in extraction.org_facts:
            confidence = f.evidence.confidence if f.evidence else None
            print(f"  Org          {f.person.name} {f.relation.value} {f.target}"
                  f" [{f.source.value}, {confidence}]")
        if extraction.proposed_terms:
            print(f"  Proposed     {', '.join(extraction.proposed_terms)}"
                  f"  (not canonical until enough tickets support them)")

        topic = args.topic or (extraction.topics[0].name if extraction.topics else None)
        if topic:
            print(f"\nWho to consider for {topic!r}:")
            for row in store.experts_for_topic(topic):
                print(f"  {row.person_name}: {row.expert_mentions} expert mention(s), "
                      f"{row.participations} ticket(s) worked")
                for excerpt in row.excerpts[:2]:
                    print(f"      “{excerpt}”")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
