"""Controlled vocabulary with a curation gate.

The requirement is that the model evolve with LLM input. Left unconstrained,
that produces drift rather than evolution: `payment gateway`, `payment-gateway`
and `billing gateway` arrive as three unrelated topics within a week, and every
routing query silently misses two thirds of the evidence.

So extraction is constrained to CANONICAL terms, and anything new is recorded
as PROPOSED with a support count. A proposal is promoted once enough
*independent tickets* back it -- one ticket mentioning a thing five times is
still one piece of evidence, so support counts distinct tickets, not mentions.

The canonical list is also what goes in the extractor's prompt. Keeping it
small and stable is what makes that prompt cacheable, which is the difference
between a cheap pipeline and an expensive one.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from graph_knowledge.models import slug

__all__ = ["TermKind", "TermStatus", "Term", "Vocabulary", "DEFAULT_PROMOTION_THRESHOLD"]

DEFAULT_PROMOTION_THRESHOLD = 3


class TermKind(str, Enum):
    TOPIC = "topic"
    FUNCTION = "function"
    TEAM = "team"


class TermStatus(str, Enum):
    CANONICAL = "canonical"   # in the prompt; extraction may use it
    PROPOSED = "proposed"     # seen, not yet trusted
    REJECTED = "rejected"     # explicitly refused; never re-proposed


class Term(BaseModel):
    kind: TermKind
    name: str
    status: TermStatus = TermStatus.PROPOSED
    #: Distinct tickets supporting this term, not total mentions.
    supporting_tickets: set[str] = Field(default_factory=set)
    #: Spellings that should resolve to this term.
    aliases: set[str] = Field(default_factory=set)

    @property
    def key(self) -> str:
        return f"term:{self.kind.value}:{slug(self.name)}"

    @property
    def support(self) -> int:
        return len(self.supporting_tickets)


class Vocabulary:
    """In-memory vocabulary. The store persists and reloads it."""

    def __init__(self, terms: list[Term] | None = None, threshold: int = DEFAULT_PROMOTION_THRESHOLD):
        self._terms: dict[str, Term] = {}
        self._threshold = threshold
        for term in terms or []:
            self._terms[term.key] = term

    # -- lookup --------------------------------------------------------------

    def resolve(self, kind: TermKind, name: str) -> Term | None:
        """Find a term by name or by any of its aliases."""
        direct = self._terms.get(f"term:{kind.value}:{slug(name)}")
        if direct is not None:
            return direct
        needle = slug(name)
        for term in self._terms.values():
            if term.kind is kind and needle in {slug(a) for a in term.aliases}:
                return term
        return None

    def canonical(self, kind: TermKind) -> list[str]:
        """The terms extraction is allowed to use. Goes into the prompt."""
        return sorted(
            t.name for t in self._terms.values()
            if t.kind is kind and t.status is TermStatus.CANONICAL
        )

    def proposed(self, kind: TermKind | None = None) -> list[Term]:
        return sorted(
            (t for t in self._terms.values()
             if t.status is TermStatus.PROPOSED and (kind is None or t.kind is kind)),
            key=lambda t: (-t.support, t.name),
        )

    def all_terms(self) -> list[Term]:
        return sorted(self._terms.values(), key=lambda t: t.key)

    # -- curation ------------------------------------------------------------

    def observe(self, kind: TermKind, name: str, ticket_id: str) -> Term:
        """Record that a ticket used a term, promoting it if it now qualifies.

        A term already CANONICAL stays canonical; a REJECTED one stays rejected
        however often it reappears, so a curation decision is not undone by
        weight of repetition.
        """
        existing = self.resolve(kind, name)
        if existing is None:
            existing = Term(kind=kind, name=name.strip())
            self._terms[existing.key] = existing

        existing.supporting_tickets.add(ticket_id)
        if existing.status is TermStatus.PROPOSED and existing.support >= self._threshold:
            existing.status = TermStatus.CANONICAL
        return existing

    def promote(self, kind: TermKind, name: str) -> Term | None:
        term = self.resolve(kind, name)
        if term is not None and term.status is not TermStatus.REJECTED:
            term.status = TermStatus.CANONICAL
        return term

    def reject(self, kind: TermKind, name: str) -> Term | None:
        term = self.resolve(kind, name)
        if term is not None:
            term.status = TermStatus.REJECTED
        return term

    def alias(self, kind: TermKind, canonical_name: str, alias: str) -> Term | None:
        """Fold a spelling into an existing term.

        The main repair tool: when the same concept has arrived under two
        names, this merges them without rewriting history.
        """
        term = self.resolve(kind, canonical_name)
        if term is None:
            return None
        duplicate = self.resolve(kind, alias)
        if duplicate is not None and duplicate.key != term.key:
            term.supporting_tickets |= duplicate.supporting_tickets
            term.aliases |= duplicate.aliases
            del self._terms[duplicate.key]
        term.aliases.add(alias)
        return term
