"""Neo4j backend.

Idempotency comes from MERGE on a deterministic `key` property rather than on
the whole property map, so re-ingesting a document updates nodes in place
instead of duplicating them.
"""

from __future__ import annotations

import os

from graph_knowledge.models import Extraction
from graph_knowledge.store.base import ActivityRow, TravelRow

__all__ = ["Neo4jStore", "neo4j_available"]

CONSTRAINTS = [
    "CREATE CONSTRAINT person_key IF NOT EXISTS FOR (n:Person) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT place_key IF NOT EXISTS FOR (n:Place) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT activity_key IF NOT EXISTS FOR (n:Activity) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT attribute_key IF NOT EXISTS FOR (n:Attribute) REQUIRE n.key IS UNIQUE",
]


def neo4j_available(uri: str | None = None, timeout: float = 3.0) -> bool:
    """True if a Bolt server answers. Used to skip integration tests."""
    try:
        from neo4j import GraphDatabase
    except ImportError:  # pragma: no cover
        return False

    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password"),
    )
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
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                user or os.environ.get("NEO4J_USER", "neo4j"),
                password or os.environ.get("NEO4J_PASSWORD", "password"),
            ),
        )

    def initialize(self) -> None:
        with self._driver.session() as session:
            for statement in CONSTRAINTS:
                session.run(statement)

    def write(self, extraction: Extraction) -> None:
        doc_id = extraction.doc_id
        with self._driver.session() as session:
            session.execute_write(self._write_tx, extraction, doc_id)

    @staticmethod
    def _write_tx(tx, extraction: Extraction, doc_id: str) -> None:
        for person in extraction.people:
            tx.run(
                """
                MERGE (p:Person {key: $key})
                SET p.name = $name,
                    p.gender = coalesce($gender, p.gender)
                """,
                key=person.key,
                name=person.name,
                gender=person.gender,
            )

        for place in extraction.places:
            tx.run(
                """
                MERGE (pl:Place {key: $key})
                SET pl.name = coalesce($name, pl.name),
                    pl.address = coalesce($address, pl.address),
                    pl.type = coalesce($type, pl.type)
                """,
                key=place.key(doc_id),
                name=place.name,
                address=place.address,
                type=place.type,
            )

        for travel in extraction.travels:
            tx.run(
                """
                MATCH (p:Person {key: $person_key})
                MATCH (pl:Place {key: $place_key})
                MERGE (p)-[t:TRAVEL]->(pl)
                SET t.datetime = coalesce($datetime, t.datetime)
                """,
                person_key=travel.person.key,
                place_key=travel.place.key(doc_id),
                datetime=travel.datetime,
            )

        for index, event in enumerate(extraction.activities):
            activity_key = event.activity_key(doc_id, index)
            tx.run(
                """
                MATCH (p:Person {key: $person_key})
                MERGE (a:Activity {key: $activity_key})
                SET a.type = $type,
                    a.datetime = coalesce($datetime, a.datetime)
                MERGE (p)-[:PERFORMED]->(a)
                """,
                person_key=event.person.key,
                activity_key=activity_key,
                type=event.activity.type,
                datetime=event.activity.datetime,
            )

            if event.place is not None:
                tx.run(
                    """
                    MATCH (a:Activity {key: $activity_key})
                    MATCH (pl:Place {key: $place_key})
                    MERGE (a)-[:AT]->(pl)
                    """,
                    activity_key=activity_key,
                    place_key=event.place.key(doc_id),
                )

            for attribute in event.activity.attributes:
                tx.run(
                    """
                    MATCH (a:Activity {key: $activity_key})
                    MERGE (at:Attribute {key: $attribute_key})
                    SET at.name = $name,
                        at.value = coalesce($value, at.value),
                        at.unit = coalesce($unit, at.unit)
                    MERGE (a)-[:HAS_ATTRIBUTE]->(at)
                    """,
                    activity_key=activity_key,
                    attribute_key=attribute.key(activity_key),
                    name=attribute.name,
                    value=attribute.value,
                    unit=attribute.unit,
                )

    def activities_for(self, person_name: str) -> list[ActivityRow]:
        query = """
        MATCH (p:Person {name: $name})-[:PERFORMED]->(a:Activity)
        OPTIONAL MATCH (a)-[:AT]->(pl:Place)
        OPTIONAL MATCH (a)-[:HAS_ATTRIBUTE]->(at:Attribute)
        RETURN p.name AS person_name, p.gender AS person_gender,
               a.type AS activity_type, a.datetime AS activity_datetime,
               pl.name AS place_name, pl.type AS place_type,
               at.name AS attribute_name, at.value AS attribute_value
        ORDER BY activity_type, attribute_name
        """
        with self._driver.session() as session:
            return [
                ActivityRow(
                    person_name=r["person_name"],
                    person_gender=r["person_gender"],
                    activity_type=r["activity_type"],
                    activity_datetime=r["activity_datetime"],
                    place_name=r["place_name"],
                    place_type=r["place_type"],
                    attribute_name=r["attribute_name"],
                    attribute_value=r["attribute_value"],
                )
                for r in session.run(query, name=person_name)
            ]

    def travels_for(self, person_name: str) -> list[TravelRow]:
        query = """
        MATCH (p:Person {name: $name})-[t:TRAVEL]->(pl:Place)
        RETURN p.name AS person_name, p.gender AS person_gender,
               t.datetime AS travel_datetime,
               pl.name AS place_name, pl.type AS place_type
        ORDER BY place_type
        """
        with self._driver.session() as session:
            return [
                TravelRow(
                    person_name=r["person_name"],
                    person_gender=r["person_gender"],
                    travel_datetime=r["travel_datetime"],
                    place_name=r["place_name"],
                    place_type=r["place_type"],
                )
                for r in session.run(query, name=person_name)
            ]

    def node_count(self) -> int:
        with self._driver.session() as session:
            return session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

    def reset(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()
