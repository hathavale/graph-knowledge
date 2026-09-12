"""Neo4j backend. Same schema and statements as the embedded store."""

from __future__ import annotations

import os

from graph_knowledge.models import TicketExtraction
from graph_knowledge.store._cypher import (
    NODE_LABELS,
    vocabulary_statements,
    write_statements,
)
from graph_knowledge.store.base import ExpertRow, OrgFactRow, TicketPersonRow
from graph_knowledge.store.embedded_store import _merge_expert_rows
from graph_knowledge.vocabulary import Term, TermKind, TermStatus, Vocabulary

__all__ = ["Neo4jStore", "neo4j_available"]

CONSTRAINTS = [
    f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
    f"FOR (n:{label}) REQUIRE n.key IS UNIQUE"
    for label in NODE_LABELS
]


def neo4j_available(uri: str | None = None, timeout: float = 3.0) -> bool:
    try:
        from neo4j import GraphDatabase
    except ImportError:  # pragma: no cover
        return False

    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "password"))
    try:
        driver = GraphDatabase.driver(uri, auth=auth, connection_timeout=timeout)
        try:
            driver.verify_connectivity()
            return True
        finally:
            driver.close()
    except Exception:
        return False


class Neo4jStore:
    def __init__(self, uri=None, user=None, password=None) -> None:
        from neo4j import GraphDatabase, NotificationDisabledClassification

        self._driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"),
                  password or os.environ.get("NEO4J_PASSWORD", "password")),
            # Properties that are null everywhere until some ticket supplies
            # one make Neo4j warn the key does not exist. Expected here;
            # PERFORMANCE and DEPRECATION notifications stay on.
            notifications_disabled_classifications=[
                NotificationDisabledClassification.UNRECOGNIZED
            ],
        )

    def initialize(self) -> None:
        with self._driver.session() as session:
            for statement in CONSTRAINTS:
                session.run(statement)

    def _run_all(self, statements) -> None:
        with self._driver.session() as session:
            def work(tx):
                for statement, params in statements:
                    tx.run(statement, **params)

            session.execute_write(work)

    def write(self, extraction: TicketExtraction) -> None:
        self._run_all(write_statements(extraction))

    def save_vocabulary(self, terms: list[Term]) -> None:
        self._run_all(vocabulary_statements(terms))

    def load_vocabulary(self) -> Vocabulary:
        with self._driver.session() as session:
            rows = session.run(
                """MATCH (v:Term)
                   RETURN v.kind AS kind, v.name AS name, v.status AS status,
                          v.supporting_tickets AS supporting_tickets,
                          v.aliases AS aliases"""
            ).data()
        return Vocabulary([
            Term(kind=TermKind(r["kind"]), name=r["name"], status=TermStatus(r["status"]),
                 supporting_tickets=set(r["supporting_tickets"] or []),
                 aliases=set(r["aliases"] or []))
            for r in rows
        ])

    def people_for_ticket(self, ticket_id: str) -> list[TicketPersonRow]:
        with self._driver.session() as session:
            participated = session.run(
                """MATCH (p:Person)-[r:PARTICIPATED_IN]->(t:Ticket {id: $id})
                   RETURN p.name AS person_name, r.role AS role ORDER BY p.name""",
                id=ticket_id,
            ).data()
            mentioned = session.run(
                """MATCH (t:Ticket {id: $id})-[m:MENTIONS]->(p:Person)
                   OPTIONAL MATCH (x:Topic) WHERE x.key = m.topic_key
                   RETURN p.name AS person_name, m.role AS role, x.name AS topic_name,
                          m.excerpt AS excerpt, m.confidence AS confidence
                   ORDER BY p.name""",
                id=ticket_id,
            ).data()
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
        with self._driver.session() as session:
            for relation, label in (("HAS_FUNCTION", "Function"), ("MEMBER_OF", "Team"),
                                    ("REPORTS_TO", "Person")):
                out += [
                    OrgFactRow(r["person_name"], relation, r["target_name"], r["source"],
                               r["confidence"], r["excerpt"])
                    for r in session.run(
                        f"""MATCH (p:Person {{name: $name}})-[r:{relation}]->(n:{label})
                            RETURN p.name AS person_name, n.name AS target_name,
                                   r.source AS source, r.confidence AS confidence,
                                   r.excerpt AS excerpt
                            ORDER BY n.name""",
                        name=person_name,
                    ).data()
                ]
        return out

    def experts_for_topic(self, topic_name: str) -> list[ExpertRow]:
        with self._driver.session() as session:
            mentions = session.run(
                """MATCH (x:Topic {name: $topic})
                   MATCH (t:Ticket)-[m:MENTIONS]->(p:Person)
                   WHERE m.topic_key = x.key AND m.role = 'expert'
                   RETURN p.name AS person_name, m.excerpt AS excerpt""",
                topic=topic_name,
            ).data()
            worked = session.run(
                """MATCH (p:Person)-[:PARTICIPATED_IN]->(t:Ticket)-[:ABOUT]->(x:Topic {name: $topic})
                   RETURN p.name AS person_name, count(t) AS n""",
                topic=topic_name,
            ).data()
        return _merge_expert_rows(topic_name, mentions, worked)

    def resolve_alias(self, alias: str) -> str | None:
        with self._driver.session() as session:
            row = session.run(
                "MATCH (a:Alias {value: $value})-[:ALIAS_OF]->(p:Person) RETURN p.name AS name",
                value=alias,
            ).single()
        return row["name"] if row else None

    def node_count(self) -> int:
        with self._driver.session() as session:
            return session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

    def reset(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()
