"""Embedded backend (LadybugDB), for tests and for running without Docker.

Same reified schema and near-identical Cypher to the Neo4j backend -- the
engine is a file on disk rather than a server, so the whole suite runs with no
infrastructure. The main difference is that node and relationship tables must
be declared up front, which is unproblematic here because reification keeps
the schema small and fixed.
"""

from __future__ import annotations

from typing import Any

from graph_knowledge.models import Extraction
from graph_knowledge.store.base import ActivityRow, TravelRow

__all__ = ["EmbeddedStore", "embedded_available"]

SCHEMA = [
    """CREATE NODE TABLE IF NOT EXISTS Person(
        key STRING, name STRING, gender STRING, PRIMARY KEY(key))""",
    """CREATE NODE TABLE IF NOT EXISTS Place(
        key STRING, name STRING, address STRING, type STRING, PRIMARY KEY(key))""",
    """CREATE NODE TABLE IF NOT EXISTS Activity(
        key STRING, type STRING, datetime TIMESTAMP, PRIMARY KEY(key))""",
    """CREATE NODE TABLE IF NOT EXISTS Attribute(
        key STRING, name STRING, value STRING, unit STRING, PRIMARY KEY(key))""",
    "CREATE REL TABLE IF NOT EXISTS TRAVEL(FROM Person TO Place, datetime TIMESTAMP)",
    "CREATE REL TABLE IF NOT EXISTS PERFORMED(FROM Person TO Activity)",
    "CREATE REL TABLE IF NOT EXISTS AT(FROM Activity TO Place)",
    "CREATE REL TABLE IF NOT EXISTS HAS_ATTRIBUTE(FROM Activity TO Attribute)",
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

        self._path = path
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

    def write(self, extraction: Extraction) -> None:
        doc_id = extraction.doc_id

        for person in extraction.people:
            self._conn.execute(
                """MERGE (p:Person {key: $key})
                   SET p.name = $name, p.gender = coalesce($gender, p.gender)""",
                {"key": person.key, "name": person.name, "gender": person.gender},
            )

        for place in extraction.places:
            self._conn.execute(
                """MERGE (pl:Place {key: $key})
                   SET pl.name = coalesce($name, pl.name),
                       pl.address = coalesce($address, pl.address),
                       pl.type = coalesce($type, pl.type)""",
                {
                    "key": place.key(doc_id),
                    "name": place.name,
                    "address": place.address,
                    "type": place.type,
                },
            )

        for travel in extraction.travels:
            self._conn.execute(
                """MATCH (p:Person {key: $person_key}), (pl:Place {key: $place_key})
                   MERGE (p)-[t:TRAVEL]->(pl)
                   SET t.datetime = coalesce($datetime, t.datetime)""",
                {
                    "person_key": travel.person.key,
                    "place_key": travel.place.key(doc_id),
                    "datetime": travel.datetime,
                },
            )

        for index, event in enumerate(extraction.activities):
            activity_key = event.activity_key(doc_id, index)
            self._conn.execute(
                """MERGE (a:Activity {key: $activity_key})
                   SET a.type = $type,
                       a.datetime = coalesce($datetime, a.datetime)""",
                {
                    "activity_key": activity_key,
                    "type": event.activity.type,
                    "datetime": event.activity.datetime,
                },
            )
            self._conn.execute(
                """MATCH (p:Person {key: $person_key}), (a:Activity {key: $activity_key})
                   MERGE (p)-[:PERFORMED]->(a)""",
                {"person_key": event.person.key, "activity_key": activity_key},
            )

            if event.place is not None:
                self._conn.execute(
                    """MATCH (a:Activity {key: $activity_key}), (pl:Place {key: $place_key})
                       MERGE (a)-[:AT]->(pl)""",
                    {
                        "activity_key": activity_key,
                        "place_key": event.place.key(doc_id),
                    },
                )

            for attribute in event.activity.attributes:
                attribute_key = attribute.key(activity_key)
                self._conn.execute(
                    """MERGE (at:Attribute {key: $attribute_key})
                       SET at.name = $name,
                           at.value = coalesce($value, at.value),
                           at.unit = coalesce($unit, at.unit)""",
                    {
                        "attribute_key": attribute_key,
                        "name": attribute.name,
                        "value": attribute.value,
                        "unit": attribute.unit,
                    },
                )
                self._conn.execute(
                    """MATCH (a:Activity {key: $activity_key}),
                             (at:Attribute {key: $attribute_key})
                       MERGE (a)-[:HAS_ATTRIBUTE]->(at)""",
                    {"activity_key": activity_key, "attribute_key": attribute_key},
                )

    def activities_for(self, person_name: str) -> list[ActivityRow]:
        rows = self._rows(
            """MATCH (p:Person {name: $name})-[:PERFORMED]->(a:Activity)
               OPTIONAL MATCH (a)-[:AT]->(pl:Place)
               OPTIONAL MATCH (a)-[:HAS_ATTRIBUTE]->(at:Attribute)
               RETURN p.name AS person_name, p.gender AS person_gender,
                      a.type AS activity_type, a.datetime AS activity_datetime,
                      pl.name AS place_name, pl.type AS place_type,
                      at.name AS attribute_name, at.value AS attribute_value
               ORDER BY a.type, at.name""",
            {"name": person_name},
        )
        return [ActivityRow(**row) for row in rows]

    def travels_for(self, person_name: str) -> list[TravelRow]:
        rows = self._rows(
            """MATCH (p:Person {name: $name})-[t:TRAVEL]->(pl:Place)
               RETURN p.name AS person_name, p.gender AS person_gender,
                      t.datetime AS travel_datetime,
                      pl.name AS place_name, pl.type AS place_type
               ORDER BY pl.type""",
            {"name": person_name},
        )
        return [TravelRow(**row) for row in rows]

    def node_count(self) -> int:
        return self._rows("MATCH (n) RETURN count(n) AS c")[0]["c"]

    def reset(self) -> None:
        self._conn.execute("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._conn.close()
