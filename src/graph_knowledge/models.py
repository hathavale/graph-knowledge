"""Domain model for a support-ticket knowledge graph.

The graph answers one question: given a ticket, who should be notified, with
what context, and who else should be kept in the loop.

Two layers, separated by **provenance** rather than by shape:

*Identity and org structure* -- who someone is, what they do, who they report
to. Authoritative when imported from a directory; currently inferred from
ticket prose, which is unreliable. Every such fact records `source` and
`confidence` so an import can later override an inference without a migration,
and so routing can refuse to escalate on a guess.

*Expertise and involvement* -- who participated in what, who was referenced as
knowing about what. Tickets are a poor source for the first layer and the only
source for this one.

Every inferred fact carries the ticket it came from and the excerpt that
supports it. Provenance is not optional here: a routing decision that cannot
say *why* someone was chosen is not reviewable, and an unreviewable notifier
gets muted.
"""

from __future__ import annotations

import datetime as dt
import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "Evidence",
    "Function",
    "Mention",
    "MentionRole",
    "OrgFact",
    "OrgRelation",
    "Participation",
    "ParticipationRole",
    "Person",
    "Source",
    "Team",
    "Ticket",
    "TicketExtraction",
    "Topic",
    "slug",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-")


class Source(str, Enum):
    """Where a fact came from. Decides whether it can be trusted or overridden."""

    IMPORTED = "imported"   # a directory export: authoritative
    INFERRED = "inferred"   # read out of ticket prose: provisional
    DERIVED = "derived"     # computed from other facts (e.g. expertise weights)


class ParticipationRole(str, Enum):
    REPORTER = "reporter"
    RESOLVER = "resolver"
    COMMENTER = "commenter"
    ASSIGNEE = "assignee"


class MentionRole(str, Enum):
    """Why a person was referenced in a ticket they did not participate in.

    This is the signal the whole product rests on: "ask Priya, she built the
    retry logic" is a claim about expertise, made by a colleague, in context.
    """

    EXPERT = "expert"                  # named as knowing about something
    ESCALATION = "escalation"          # named as who to escalate to
    APPROVER = "approver"              # named as needing to sign off
    AFFECTED = "affected"              # named as impacted
    UNSPECIFIED = "unspecified"


class Evidence(BaseModel):
    """Why we believe a fact. Attached to every inferred edge."""

    ticket_id: str
    excerpt: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    @field_validator("excerpt")
    @classmethod
    def _trim(cls, v: str) -> str:
        # Long excerpts blow up both storage and the context packs built from
        # them; a sentence is enough to justify a fact to a human.
        return v.strip()[:300]


class Person(BaseModel):
    """Someone referenced in or participating in a ticket.

    Without a directory, identity is inferred. `aliases` carries every spelling
    seen -- "@priya", "priya@co.com", "Priya N" -- so a later mention can
    resolve to an existing person instead of creating a duplicate.
    """

    name: str
    aliases: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("person name must not be empty")
        return v.strip()

    @property
    def key(self) -> str:
        return f"person:{slug(self.name)}"

    def alias_keys(self) -> list[str]:
        return [f"alias:{slug(a)}" for a in self.aliases if a.strip()]


class Function(BaseModel):
    """A job function: Support Engineer, Product Manager, SRE."""

    name: str

    @property
    def key(self) -> str:
        return f"function:{slug(self.name)}"


class Team(BaseModel):
    name: str

    @property
    def key(self) -> str:
        return f"team:{slug(self.name)}"


class Topic(BaseModel):
    """What a ticket is about. Drawn from the controlled vocabulary."""

    name: str

    @property
    def key(self) -> str:
        return f"topic:{slug(self.name)}"


class OrgRelation(str, Enum):
    REPORTS_TO = "REPORTS_TO"
    MEMBER_OF = "MEMBER_OF"
    HAS_FUNCTION = "HAS_FUNCTION"


class OrgFact(BaseModel):
    """A structural claim about a person.

    `target` is a person name for REPORTS_TO, a team name for MEMBER_OF, a
    function name for HAS_FUNCTION. Kept deliberately low-trust: inferred
    reporting lines are the least reliable thing in this graph, and routing
    should treat them as a hint, never as authority to escalate.
    """

    person: Person
    relation: OrgRelation
    target: str
    source: Source = Source.INFERRED
    evidence: Evidence | None = None

    def target_key(self) -> str:
        if self.relation is OrgRelation.REPORTS_TO:
            return f"person:{slug(self.target)}"
        if self.relation is OrgRelation.MEMBER_OF:
            return f"team:{slug(self.target)}"
        return f"function:{slug(self.target)}"


class Participation(BaseModel):
    """Someone acted on this ticket. The strongest expertise signal there is."""

    person: Person
    role: ParticipationRole
    at: dt.datetime | None = None


class Mention(BaseModel):
    """Someone was referenced in this ticket without necessarily acting on it."""

    person: Person
    role: MentionRole = MentionRole.UNSPECIFIED
    topic: Topic | None = None   # what they were named as knowing about
    evidence: Evidence | None = None


class Ticket(BaseModel):
    id: str
    title: str | None = None
    opened_at: dt.datetime | None = None

    @property
    def key(self) -> str:
        return f"ticket:{slug(self.id)}"


class TicketExtraction(BaseModel):
    """Everything one ticket yielded. The unit of work for a write."""

    ticket: Ticket
    text: str
    people: list[Person] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    participations: list[Participation] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)
    org_facts: list[OrgFact] = Field(default_factory=list)
    #: Terms the extractor proposed that are not yet in the vocabulary.
    proposed_terms: list[str] = Field(default_factory=list)
