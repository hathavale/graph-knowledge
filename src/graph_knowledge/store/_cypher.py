"""Cypher shared by both backends.

The two engines accept near-identical Cypher for this schema, so the write
path lives here once. Duplicating it was tolerable for four node types; at
this size it would guarantee the backends drift apart silently.

Each builder returns (statement, parameters) pairs in dependency order --
nodes before the relationships that MATCH them.
"""

from __future__ import annotations

from typing import Any, Iterable

from graph_knowledge.models import Source, TicketExtraction
from graph_knowledge.vocabulary import Term

__all__ = ["write_statements", "vocabulary_statements", "NODE_LABELS"]

NODE_LABELS = ("Person", "Alias", "Ticket", "Topic", "Function", "Team", "Term")

Statement = tuple[str, dict[str, Any]]


def _evidence(fact: Any) -> dict[str, Any]:
    """Flatten Evidence onto edge properties.

    Neither engine stores nested maps, and provenance belongs on the edge it
    justifies rather than in a side table -- a routing decision has to be able
    to quote its reason without a second lookup.
    """
    evidence = getattr(fact, "evidence", None)
    if evidence is None:
        return {"ticket_id": None, "excerpt": None, "confidence": None}
    return {
        "ticket_id": evidence.ticket_id,
        "excerpt": evidence.excerpt,
        "confidence": evidence.confidence,
    }


def write_statements(extraction: TicketExtraction) -> list[Statement]:
    out: list[Statement] = []
    ticket = extraction.ticket

    out.append((
        """MERGE (t:Ticket {key: $key})
           SET t.id = $id,
               t.title = coalesce($title, t.title),
               t.opened_at = coalesce($opened_at, t.opened_at)""",
        {"key": ticket.key, "id": ticket.id, "title": ticket.title,
         "opened_at": ticket.opened_at},
    ))

    for person in extraction.people:
        out.append((
            "MERGE (p:Person {key: $key}) SET p.name = $name",
            {"key": person.key, "name": person.name},
        ))
        # Aliases are their own nodes so a later mention of "@priya" resolves
        # to an existing person instead of minting a duplicate.
        for alias_key, alias_value in zip(person.alias_keys(), person.aliases):
            out.append((
                "MERGE (a:Alias {key: $key}) SET a.value = $value",
                {"key": alias_key, "value": alias_value},
            ))
            out.append((
                """MATCH (a:Alias {key: $alias_key}), (p:Person {key: $person_key})
                   MERGE (a)-[:ALIAS_OF]->(p)""",
                {"alias_key": alias_key, "person_key": person.key},
            ))

    for topic in extraction.topics:
        out.append((
            "MERGE (x:Topic {key: $key}) SET x.name = $name",
            {"key": topic.key, "name": topic.name},
        ))
        out.append((
            """MATCH (t:Ticket {key: $ticket_key}), (x:Topic {key: $topic_key})
               MERGE (t)-[:ABOUT]->(x)""",
            {"ticket_key": ticket.key, "topic_key": topic.key},
        ))

    for participation in extraction.participations:
        out.append((
            """MATCH (p:Person {key: $person_key}), (t:Ticket {key: $ticket_key})
               MERGE (p)-[r:PARTICIPATED_IN]->(t)
               SET r.role = $role, r.at = coalesce($at, r.at)""",
            {"person_key": participation.person.key, "ticket_key": ticket.key,
             "role": participation.role.value, "at": participation.at},
        ))

    for mention in extraction.mentions:
        evidence = _evidence(mention)
        out.append((
            """MATCH (t:Ticket {key: $ticket_key}), (p:Person {key: $person_key})
               MERGE (t)-[m:MENTIONS]->(p)
               SET m.role = $role, m.topic_key = $topic_key,
                   m.excerpt = $excerpt, m.confidence = $confidence""",
            {"ticket_key": ticket.key, "person_key": mention.person.key,
             "role": mention.role.value,
             "topic_key": mention.topic.key if mention.topic else None,
             "excerpt": evidence["excerpt"], "confidence": evidence["confidence"]},
        ))

    for fact in extraction.org_facts:
        evidence = _evidence(fact)
        target_key = fact.target_key()
        label = {"REPORTS_TO": "Person", "MEMBER_OF": "Team",
                 "HAS_FUNCTION": "Function"}[fact.relation.value]
        if label != "Person":
            out.append((
                f"MERGE (n:{label} {{key: $key}}) SET n.name = $name",
                {"key": target_key, "name": fact.target},
            ))
        else:
            out.append((
                "MERGE (n:Person {key: $key}) SET n.name = coalesce(n.name, $name)",
                {"key": target_key, "name": fact.target},
            ))
        # An imported fact must never be downgraded by a later inference, so
        # `source` is only overwritten when the incoming fact is imported.
        out.append((
            f"""MATCH (p:Person {{key: $person_key}}), (n:{label} {{key: $target_key}})
                MERGE (p)-[r:{fact.relation.value}]->(n)
                SET r.source = CASE WHEN r.source = '{Source.IMPORTED.value}'
                                    THEN r.source ELSE $source END,
                    r.confidence = $confidence,
                    r.ticket_id = $ticket_id,
                    r.excerpt = $excerpt""",
            {"person_key": fact.person.key, "target_key": target_key,
             "source": fact.source.value, **evidence},
        ))

    return out


def vocabulary_statements(terms: Iterable[Term]) -> list[Statement]:
    return [(
        """MERGE (v:Term {key: $key})
           SET v.kind = $kind, v.name = $name, v.status = $status,
               v.supporting_tickets = $supporting_tickets, v.aliases = $aliases""",
        {"key": term.key, "kind": term.kind.value, "name": term.name,
         "status": term.status.value,
         "supporting_tickets": sorted(term.supporting_tickets),
         "aliases": sorted(term.aliases)},
    ) for term in terms]
