"""Domain model for the knowledge graph.

Activities are *reified*: an activity is a node, not an edge. Two reasons --
neither Neo4j nor the embedded engine supports nested map properties, and an
edge cannot itself carry edges, so an activity modelled as an edge has nowhere
to hang its attributes, participants or provenance.

    (:Person)-[:PERFORMED]->(:Activity)-[:AT]->(:Place)
    (:Activity)-[:HAS_ATTRIBUTE]->(:Attribute)

Unknown values are `None`, which both engines store as null. A property set to
null is indistinguishable from an absent property, so "known to be unknown" is
expressed by an Attribute node that exists with no `value` -- which is exactly
what reification buys.
"""

from __future__ import annotations

import re
import datetime as dt

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "Activity",
    "ActivityEvent",
    "Attribute",
    "Extraction",
    "Person",
    "Place",
    "Travel",
    "slug",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(value: str) -> str:
    """Normalise a string into a stable key fragment."""
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-")


class Attribute(BaseModel):
    """A named property of an activity, whose value may be unknown.

    "withdrew some money" yields Attribute(name="amount", value=None): we know
    an amount exists, we do not know what it was.
    """

    name: str
    value: str | None = None
    unit: str | None = None

    def key(self, activity_key: str) -> str:
        return f"{activity_key}:attr:{slug(self.name)}"


class Person(BaseModel):
    name: str
    gender: str | None = None

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("person name must not be empty")
        return v.strip()

    @property
    def key(self) -> str:
        # People are identified by name globally, so the same person mentioned
        # in two documents merges into one node.
        return f"person:{slug(self.name)}"


class Place(BaseModel):
    """A location. Any of name/address/type may be unknown.

    Keying deserves care. A *named* place is global: "Chase Bank" in two
    documents is one node. An *unnamed* place ("the bank") is scoped to its
    document, so the two mentions in a single passage unify into one node while
    an unrelated document's unnamed bank stays distinct. Without that scoping,
    every anonymous bank in the corpus would silently collapse into one.
    """

    name: str | None = None
    address: str | None = None
    type: str | None = None

    def key(self, doc_id: str) -> str:
        if self.name:
            return f"place:{slug(self.name)}"
        return f"place:doc:{slug(doc_id)}:{slug(self.type or 'unknown')}"


class Activity(BaseModel):
    type: str
    datetime: dt.datetime | None = None
    attributes: list[Attribute] = Field(default_factory=list)


class Travel(BaseModel):
    """Person moved to a place. A plain edge -- it has no sub-structure."""

    person: Person
    place: Place
    datetime: dt.datetime | None = None


class ActivityEvent(BaseModel):
    """Person performed an activity, optionally at a place."""

    person: Person
    activity: Activity
    place: Place | None = None

    def activity_key(self, doc_id: str, index: int) -> str:
        # Activities are events: always document-scoped, never merged across
        # documents even when the type string is identical.
        return f"activity:doc:{slug(doc_id)}:{index}:{slug(self.activity.type)}"


class Extraction(BaseModel):
    """Everything one document yielded. The unit of work for a write."""

    doc_id: str
    text: str
    people: list[Person] = Field(default_factory=list)
    places: list[Place] = Field(default_factory=list)
    travels: list[Travel] = Field(default_factory=list)
    activities: list[ActivityEvent] = Field(default_factory=list)
