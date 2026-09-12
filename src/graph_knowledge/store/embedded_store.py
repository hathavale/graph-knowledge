"""Embedded backend (LadybugDB). No server, so tests need no infrastructure."""

from __future__ import annotations

from typing import Any

from graph_knowledge.models import TicketExtraction
from graph_knowledge.store._cypher import write_statements, vocabulary_statements
from graph_knowledge.store.base import ExpertRow, OrgFactRow, TicketPersonRow
from graph_knowledge.vocabulary import Term, TermKind, TermStatus, Vocabulary

__all__ = ["EmbeddedStore", "embedded_available"]

SCHEMA = [
    "CREATE NODE TABLE IF NOT EXISTS Person(key STRING, name STRING, PRIMARY KEY(key))",
    "CREATE NODE TABLE IF NOT EXISTS Alias(key STRING, value STRING, PRIMARY KEY(key))",
    """CREATE NODE TABLE IF NOT EXISTS Ticket(
        key STRING, id STRING, title STRING, opened_at TIMESTAMP, PRIMARY KEY(key))""",
    "CREATE NODE TABLE IF NOT EXISTS Topic(key STRING, name STRING, PRIMARY KEY(key))",
    "CREATE NODE TABLE IF NOT EXISTS Function(key STRING, name STRING, PRIMARY KEY(key))",
    "CREATE NODE TABLE IF NOT EXISTS Team(key STRING, name STRING, PRIMARY KEY(key))",
    """CREATE NODE TABLE IF NOT EXISTS Term(
        key STRING, kind STRING, name STRING, status STRING,
        supporting_tickets STRING[], aliases STRING[], PRIMARY KEY(key))""",
    "CREATE REL TABLE IF NOT EXISTS ALIAS_OF(FROM Alias TO Person)",
    "CREATE REL TABLE IF NOT EXISTS ABOUT(FROM Ticket TO Topic)",
    "CREATE REL TABLE IF NOT EXISTS PARTICIPATED_IN(FROM Person TO Ticket, role STRING, at TIMESTAMP)",
    """CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Ticket TO Person,
        role STRING, topic_key STRING, excerpt STRING, confidence DOUBLE)""",
    """CREATE REL TABLE IF NOT EXISTS HAS_FUNCTION(FROM Person TO Function,
        source STRING, confidence DOUBLE, ticket_id STRING, excerpt STRING)""",
    """CREATE REL TABLE IF NOT EXISTS MEMBER_OF(FROM Person TO Team,
        source STRING, confidence DOUBLE, ticket_id STRING, excerpt STRING)""",
    """CREATE REL TABLE IF NOT EXISTS REPORTS_TO(FROM Person TO Person,
        source STRING, confidence DOUBLE, ticket_id STRING, excerpt STRING)""",
]


def embedded_available() -> bool:
    try:
        import ladybug  # noqa: F401
    except ImportError:
        return False
    return True


class EmbeddedStore:
    def __init__(self, path: str) -> None:
        import ladybug

        self._db = ladybug.Database(path)
        self._conn = ladybug.Connection(self._db)

    def initialize(self) -> None:
        for statement in SCHEMA:
            self._conn.execute(statement)

    def _rows(self, query: str, params: dict[str, Any] | None = None) -> list[dict]:
        result = self._conn.execute(query, params or {})
        columns = result.get_column_names()
        rows = []
        while result.has_next():
            rows.append(dict(zip(columns, result.get_next())))
        return rows

    def write(self, extraction: TicketExtraction) -> None:
        for statement, params in write_statements(extraction):
            self._conn.execute(statement, params)

    def save_vocabulary(self, terms: list[Term]) -> None:
        for statement, params in vocabulary_statements(terms):
            self._conn.execute(statement, params)

    def load_vocabulary(self) -> Vocabulary:
        rows = self._rows(
            """MATCH (v:Term)
               RETURN v.kind AS kind, v.name AS name, v.status AS status,
                      v.supporting_tickets AS supporting_tickets, v.aliases AS aliases"""
        )
        return Vocabulary([
            Term(
                kind=TermKind(r["kind"]), name=r["name"], status=TermStatus(r["status"]),
                supporting_tickets=set(r["supporting_tickets"] or []),
                aliases=set(r["aliases"] or []),
            )
            for r in rows
        ])

    def people_for_ticket(self, ticket_id: str) -> list[TicketPersonRow]:
        participated = self._rows(
            """MATCH (p:Person)-[r:PARTICIPATED_IN]->(t:Ticket {id: $id})
               RETURN p.name AS person_name, r.role AS role
               ORDER BY p.name""",
            {"id": ticket_id},
        )
        mentioned = self._rows(
            """MATCH (t:Ticket {id: $id})-[m:MENTIONS]->(p:Person)
               OPTIONAL MATCH (x:Topic) WHERE x.key = m.topic_key
               RETURN p.name AS person_name, m.role AS role, x.name AS topic_name,
                      m.excerpt AS excerpt, m.confidence AS confidence
               ORDER BY p.name""",
            {"id": ticket_id},
        )
        return [
            TicketPersonRow(r["person_name"], "participated", r["role"], None, None, None)
            for r in participated
        ] + [
            TicketPersonRow(r["person_name"], "mentioned", r["role"], r["topic_name"],
                            r["excerpt"], r["confidence"])
            for r in mentioned
        ]

    def org_facts_for(self, person_name: str) -> list[OrgFactRow]:
        out: list[OrgFactRow] = []
        for relation, label in (("HAS_FUNCTION", "Function"), ("MEMBER_OF", "Team"),
                                ("REPORTS_TO", "Person")):
            out += [
                OrgFactRow(r["person_name"], relation, r["target_name"], r["source"],
                           r["confidence"], r["excerpt"])
                for r in self._rows(
                    f"""MATCH (p:Person {{name: $name}})-[r:{relation}]->(n:{label})
                        RETURN p.name AS person_name, n.name AS target_name,
                               r.source AS source, r.confidence AS confidence,
                               r.excerpt AS excerpt
                        ORDER BY n.name""",
                    {"name": person_name},
                )
            ]
        return out

    def experts_for_topic(self, topic_name: str) -> list[ExpertRow]:
        """Candidates ranked by evidence.

        Two signals, counted separately because they mean different things: a
        colleague naming someone as the expert is a deliberate claim, while
        having worked the ticket is behavioural. Neither is folded into a
        single opaque score here -- ranking policy belongs in the routing
        layer, on top of these counts.
        """
        mentions = self._rows(
            """MATCH (x:Topic {name: $topic})
               MATCH (t:Ticket)-[m:MENTIONS]->(p:Person)
               WHERE m.topic_key = x.key AND m.role = 'expert'
               RETURN p.name AS person_name, m.excerpt AS excerpt""",
            {"topic": topic_name},
        )
        worked = self._rows(
            """MATCH (p:Person)-[:PARTICIPATED_IN]->(t:Ticket)-[:ABOUT]->(x:Topic {name: $topic})
               RETURN p.name AS person_name, count(t) AS n""",
            {"topic": topic_name},
        )
        return _merge_expert_rows(topic_name, mentions, worked)

    def resolve_alias(self, alias: str) -> str | None:
        rows = self._rows(
            "MATCH (a:Alias {value: $value})-[:ALIAS_OF]->(p:Person) RETURN p.name AS name",
            {"value": alias},
        )
        return rows[0]["name"] if rows else None

    def node_count(self) -> int:
        return self._rows("MATCH (n) RETURN count(n) AS c")[0]["c"]

    def reset(self) -> None:
        self._conn.execute("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._conn.close()


def _merge_expert_rows(topic: str, mentions: list[dict], worked: list[dict]) -> list[ExpertRow]:
    tally: dict[str, dict] = {}
    for row in mentions:
        entry = tally.setdefault(row["person_name"], {"m": 0, "p": 0, "e": []})
        entry["m"] += 1
        if row.get("excerpt"):
            entry["e"].append(row["excerpt"])
    for row in worked:
        entry = tally.setdefault(row["person_name"], {"m": 0, "p": 0, "e": []})
        entry["p"] += row["n"]
    return sorted(
        (ExpertRow(name, topic, v["p"], v["m"], v["e"]) for name, v in tally.items()),
        key=lambda r: (-(r.expert_mentions * 2 + r.participations), r.person_name),
    )
