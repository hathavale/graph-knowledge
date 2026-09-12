"""The storage seam.

Both backends speak Cypher and share the reified schema, so the engine choice
stays reversible: the same queries and the same tests run against either.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from graph_knowledge.models import Extraction

__all__ = ["ActivityRow", "GraphStore", "TravelRow"]


@dataclass(frozen=True)
class ActivityRow:
    person_name: str
    person_gender: str | None
    activity_type: str
    activity_datetime: datetime | None
    place_name: str | None
    place_type: str | None
    attribute_name: str | None
    attribute_value: str | None


@dataclass(frozen=True)
class TravelRow:
    person_name: str
    person_gender: str | None
    travel_datetime: datetime | None
    place_name: str | None
    place_type: str | None


@runtime_checkable
class GraphStore(Protocol):
    def initialize(self) -> None:
        """Create schema/constraints. Must be safe to call repeatedly."""
        ...

    def write(self, extraction: Extraction) -> None:
        """Persist an extraction. Idempotent: re-writing changes nothing."""
        ...

    def activities_for(self, person_name: str) -> list[ActivityRow]: ...

    def travels_for(self, person_name: str) -> list[TravelRow]: ...

    def node_count(self) -> int: ...

    def reset(self) -> None:
        """Delete all data. For tests and local resets."""
        ...

    def close(self) -> None: ...
