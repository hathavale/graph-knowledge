"""The storage seam.

Both backends speak the same Cypher (see `_cypher.py`) over the same schema,
so the engine choice stays reversible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from graph_knowledge.models import TicketExtraction
from graph_knowledge.vocabulary import Term, Vocabulary

__all__ = ["ExpertRow", "GraphStore", "OrgFactRow", "TicketPersonRow"]


@dataclass(frozen=True)
class TicketPersonRow:
    person_name: str
    relation: str          # "participated" | "mentioned"
    role: str
    topic_name: str | None
    excerpt: str | None
    confidence: float | None


@dataclass(frozen=True)
class OrgFactRow:
    person_name: str
    relation: str
    target_name: str
    source: str
    confidence: float | None
    excerpt: str | None


@dataclass(frozen=True)
class ExpertRow:
    """A candidate to notify, with the evidence that nominated them."""

    person_name: str
    topic_name: str
    participations: int
    expert_mentions: int
    excerpts: list[str]


@runtime_checkable
class GraphStore(Protocol):
    def initialize(self) -> None: ...

    def write(self, extraction: TicketExtraction) -> None:
        """Persist one ticket. Idempotent: re-writing changes nothing."""
        ...

    def save_vocabulary(self, terms: list[Term]) -> None: ...

    def load_vocabulary(self) -> Vocabulary: ...

    def people_for_ticket(self, ticket_id: str) -> list[TicketPersonRow]: ...

    def org_facts_for(self, person_name: str) -> list[OrgFactRow]: ...

    def experts_for_topic(self, topic_name: str) -> list[ExpertRow]:
        """Who to consider notifying about a topic, and why."""
        ...

    def resolve_alias(self, alias: str) -> str | None:
        """Map a handle or address to a canonical person name."""
        ...

    def node_count(self) -> int: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
