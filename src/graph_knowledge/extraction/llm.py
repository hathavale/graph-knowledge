"""LLM extraction of ticket facts, validated against a schema.

Constrained by the controlled vocabulary: the canonical topic and function
lists go into the prompt, and anything outside them is returned separately as
a *proposal* rather than written straight into the graph. That is what lets
the model evolve without drifting -- see `vocabulary.py`.

As before, the model emits a flat document with local ids and we check
referential integrity ourselves, so a mention pointing at a person the model
never declared raises rather than being persisted.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from graph_knowledge.models import (
    Evidence,
    Mention,
    MentionRole,
    OrgFact,
    OrgRelation,
    Participation,
    ParticipationRole,
    Person,
    Source,
    Ticket,
    TicketExtraction,
    Topic,
)
from graph_knowledge.vocabulary import TermKind, Vocabulary

__all__ = ["ExtractionError", "LLMExtractor", "DEFAULT_MODEL"]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000


class ExtractionError(RuntimeError):
    """The model's output could not be turned into a valid extraction."""


# --- model-facing schema ----------------------------------------------------
# No defaults on optional fields: structured outputs requires every property to
# be present, so the model must emit an explicit null rather than omit a key.


class _LLMPerson(BaseModel):
    id: str
    name: str
    aliases: list[str]


class _LLMTopic(BaseModel):
    id: str
    name: str
    is_new: bool          # not in the canonical list the prompt supplied


class _LLMParticipation(BaseModel):
    person_id: str
    role: str
    at: str | None


class _LLMMention(BaseModel):
    person_id: str
    role: str
    topic_id: str | None
    excerpt: str
    confidence: float


class _LLMOrgFact(BaseModel):
    person_id: str
    relation: str
    target: str
    excerpt: str
    confidence: float


class _LLMTicketExtraction(BaseModel):
    people: list[_LLMPerson]
    topics: list[_LLMTopic]
    participations: list[_LLMParticipation]
    mentions: list[_LLMMention]
    org_facts: list[_LLMOrgFact]


SYSTEM_TEMPLATE = """\
You extract a knowledge graph from support tickets. Its purpose is to decide,
for a future ticket, who should be notified and why -- so every fact must be
traceable to something the ticket actually says.

Give each person and topic a short lowercase id unique within this ticket, and
refer to them by those ids.

PEOPLE
- Emit a person only for a specific, identifiable individual. Handles
  (@priya), email addresses and names all count. Do NOT create a person for an
  unnamed role like "the on-call engineer" or "support".
- Resolve every reference to one person: "@priya", "Priya", "priya@co.com" and
  "she" in the same ticket are one entry. Put every spelling seen in `aliases`
  and use the fullest form as `name`.

PARTICIPATION vs MENTION -- the distinction matters most in this task.
- A *participation* is someone acting on this ticket: reporting, commenting,
  being assigned, resolving. Roles: reporter, assignee, commenter, resolver.
- A *mention* is someone referenced without necessarily acting. Roles:
    expert       -- named as knowing about something ("ask Priya, she built it")
    escalation   -- named as who to escalate to
    approver     -- named as needing to sign off
    affected     -- named as impacted
    unspecified  -- referenced with no clear purpose
- Someone can be both. Emit both entries.

TOPICS
- Use a topic from this list wherever it fits, spelled exactly as given, with
  is_new = false:
{topics}
- If the ticket is clearly about something not in that list, emit it with
  is_new = true. Prefer an existing topic over a near-duplicate new one:
  propose a new topic only when nothing listed covers it.

ORG FACTS -- be conservative; these are the least reliable thing you produce.
- relation is one of: REPORTS_TO (target: a person's name), MEMBER_OF (target:
  a team name), HAS_FUNCTION (target: a job function).
- Emit one only where the ticket states or plainly implies it ("Priya's
  manager, Dev", "I'm the PM for billing"). Never guess a function from what
  someone did, and never guess a reporting line from who answered whom.
- Known functions: {functions}. Use one of these when it fits; otherwise use
  the ticket's own wording.

EVIDENCE
- Every mention and org fact needs `excerpt`: the shortest span of the ticket
  that supports it, quoted verbatim. A fact whose excerpt does not support it
  is worse than a missing fact, because a person will be messaged on it.
- `confidence` is 0.0-1.0. Use below 0.5 when you are inferring rather than
  reading, and prefer omitting a fact entirely to asserting a weak one.

TIMES
- ISO 8601 when the ticket gives an absolute date or time. Never resolve a
  relative expression ("yesterday", "last week") -- use null.
"""


def _parse_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("discarding unparseable datetime %r", value)
        return None


def _enum(cls, value: str, default):
    try:
        return cls(value.strip().lower())
    except ValueError:
        logger.warning("unknown %s %r, using %s", cls.__name__, value, default)
        return default


class LLMExtractor:
    def __init__(
        self,
        vocabulary: Vocabulary | None = None,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str | None = None,
    ) -> None:
        self._vocabulary = vocabulary or Vocabulary()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self.last_response: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            # The SDK retries 429/5xx with backoff; no hand-rolled loop here.
            self._client = anthropic.Anthropic()
        return self._client

    def system_prompt(self) -> str:
        topics = self._vocabulary.canonical(TermKind.TOPIC)
        functions = self._vocabulary.canonical(TermKind.FUNCTION)
        return SYSTEM_TEMPLATE.format(
            topics="\n".join(f"  - {t}" for t in topics) or "  (none yet)",
            functions=", ".join(functions) or "none recorded yet",
        )

    def extract(self, text: str, ticket: Ticket | str) -> TicketExtraction:
        if isinstance(ticket, str):
            ticket = Ticket(id=ticket)
        response = self._call(text)
        self.last_response = response
        raw = getattr(response, "parsed_output", None)
        if raw is None:
            raise ExtractionError(
                f"model returned no parsed output for {ticket.id!r} "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r})"
            )
        return self._to_domain(raw, text=text, ticket=ticket)

    def _call(self, text: str) -> Any:
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            # Stable across tickets within a vocabulary generation, so it
            # caches. Promoting a term invalidates it, which is the cost of
            # letting the vocabulary evolve -- it settles as the set matures.
            "system": [{"type": "text", "text": self.system_prompt(),
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": text}],
            "output_format": _LLMTicketExtraction,
            "thinking": {"type": "adaptive"},
        }
        if self._effort is not None:
            kwargs["output_config"] = {"effort": self._effort}

        try:
            return self.client.messages.parse(**kwargs)
        except anthropic.NotFoundError as exc:
            raise ExtractionError(f"unknown model {self._model!r}") from exc
        except anthropic.RateLimitError as exc:
            raise ExtractionError("rate limited after SDK retries") from exc
        except anthropic.APIStatusError as exc:
            raise ExtractionError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ExtractionError("could not reach the API") from exc
        except ValidationError as exc:
            raise ExtractionError(f"response did not match the schema: {exc}") from exc
        except TypeError as exc:
            if "authentication" in str(exc).lower():
                raise ExtractionError(
                    "no API credentials found -- set ANTHROPIC_API_KEY or run `ant auth login`"
                ) from exc
            raise

    def _to_domain(
        self, raw: _LLMTicketExtraction, text: str, ticket: Ticket
    ) -> TicketExtraction:
        people: dict[str, Person] = {}
        for item in raw.people:
            if not item.name.strip():
                logger.warning("skipping person %r with empty name", item.id)
                continue
            people[item.id] = Person(name=item.name, aliases=item.aliases)

        topics: dict[str, Topic] = {}
        proposed: list[str] = []
        for item in raw.topics:
            existing = self._vocabulary.resolve(TermKind.TOPIC, item.name)
            # Fold a proposal onto its canonical spelling when one exists, so
            # an alias does not enter the graph as a second topic.
            topics[item.id] = Topic(name=existing.name if existing else item.name)
            if item.is_new and existing is None:
                proposed.append(item.name)

        def person(ref: str, context: str) -> Person:
            try:
                return people[ref]
            except KeyError:
                raise ExtractionError(
                    f"{context} references undeclared person id {ref!r} "
                    f"(declared: {sorted(people)})"
                ) from None

        participations = [
            Participation(
                person=person(p.person_id, "participation"),
                role=_enum(ParticipationRole, p.role, ParticipationRole.COMMENTER),
                at=_parse_datetime(p.at),
            )
            for p in raw.participations
        ]

        mentions = []
        for m in raw.mentions:
            topic = topics.get(m.topic_id) if m.topic_id else None
            if m.topic_id and topic is None:
                raise ExtractionError(
                    f"mention references undeclared topic id {m.topic_id!r} "
                    f"(declared: {sorted(topics)})"
                )
            mentions.append(Mention(
                person=person(m.person_id, "mention"),
                role=_enum(MentionRole, m.role, MentionRole.UNSPECIFIED),
                topic=topic,
                evidence=Evidence(ticket_id=ticket.id, excerpt=m.excerpt,
                                  confidence=m.confidence),
            ))

        org_facts = []
        for f in raw.org_facts:
            try:
                relation = OrgRelation(f.relation.strip().upper())
            except ValueError:
                logger.warning("dropping org fact with unknown relation %r", f.relation)
                continue
            org_facts.append(OrgFact(
                person=person(f.person_id, "org fact"),
                relation=relation,
                target=f.target,
                source=Source.INFERRED,
                evidence=Evidence(ticket_id=ticket.id, excerpt=f.excerpt,
                                  confidence=f.confidence),
            ))

        return TicketExtraction(
            ticket=ticket, text=text,
            people=list(people.values()),
            topics=list(topics.values()),
            participations=participations,
            mentions=mentions,
            org_facts=org_facts,
            proposed_terms=proposed,
        )
