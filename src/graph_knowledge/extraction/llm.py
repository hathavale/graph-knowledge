"""LLM-backed extractor.

Uses structured outputs (`client.messages.parse`) so the model's response is
validated against a Pydantic schema before it reaches the graph. The schema is
the validation boundary: a malformed extraction raises here rather than
quietly writing nonsense into the store.

The model emits a *flat* document with local string ids rather than the nested
domain model. Two reasons: a nested shape would make the model repeat each
person inside every travel and activity (inviting "Mary" with a gender in one
place and without in another), and flat ids let us verify referential
integrity ourselves -- an activity pointing at a person the model never
declared is a bug we want to catch, not persist.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from graph_knowledge.models import (
    Activity,
    ActivityEvent,
    Attribute,
    Extraction,
    Person,
    Place,
    Travel,
)

__all__ = ["ExtractionError", "LLMExtractor", "DEFAULT_MODEL"]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000


class ExtractionError(RuntimeError):
    """The model's output could not be turned into a valid Extraction."""


# --- the model-facing schema ------------------------------------------------
# Optional fields carry no defaults on purpose: structured outputs requires
# every property to be present, so the model must emit an explicit null rather
# than omit the key. That distinction is the whole point of this pipeline.


class _LLMPerson(BaseModel):
    id: str
    name: str
    gender: str | None


class _LLMPlace(BaseModel):
    id: str
    name: str | None
    address: str | None
    type: str | None


class _LLMAttribute(BaseModel):
    name: str
    value: str | None
    unit: str | None


class _LLMTravel(BaseModel):
    person_id: str
    place_id: str
    datetime: str | None


class _LLMActivity(BaseModel):
    person_id: str
    place_id: str | None
    type: str
    datetime: str | None
    attributes: list[_LLMAttribute]


class _LLMExtraction(BaseModel):
    people: list[_LLMPerson]
    places: list[_LLMPlace]
    travels: list[_LLMTravel]
    activities: list[_LLMActivity]


SYSTEM_PROMPT = """\
You extract a knowledge graph of people, places and activities from text.

Return people, places, travels and activities. Give every person and place a
short lowercase id unique within this document (e.g. "mary", "bank_1"); refer
to them from travels and activities by those ids only.

Rules:

1. Coreference. Resolve pronouns and descriptions to the entity they refer to.
   "Mary went to the bank. She withdrew some money." is two statements about
   one person, so emit ONE person with id "mary" and have both the travel and
   the activity reference it.

2. Gender. Set gender only when the text evidences it -- a gendered pronoun
   referring to the person, an explicit statement, or a gendered noun such as
   "her sister". Use "Female", "Male", or "Other". Never infer gender from a
   first name alone. Otherwise null.

3. Unknowns are null. Never invent a name, address, date or amount that the
   text does not state. An unnamed place still gets an entry with name null
   and a type, if the type is stated ("the bank" -> type "Bank").

4. Repeated mentions of the same unnamed place within this document are the
   SAME place and must share one id.

5. Activity type is "<base verb> <object>", lowercase, with the verb in its
   base form: "withdrew some money" -> "withdraw money"; "bought a car" ->
   "buy car". Normalise currency words to "money".

6. Attributes record properties of an activity. When an activity implies a
   property the text does not quantify, emit the attribute with a null value:
   "withdrew some money" -> attribute name "amount", value null. When the text
   does state it, fill it in: "withdrew 500 dollars" -> name "amount",
   value "500", unit "dollars".

7. Datetimes are ISO 8601 strings when the text gives an absolute date or
   time. Do not resolve relative expressions ("last Tuesday") into a date --
   use null.

8. place_id on an activity is where it happened, when the text supports it --
   including a place established by a preceding sentence. Otherwise null.
"""


def _parse_datetime(value: str | None) -> dt.datetime | None:
    """ISO 8601 or nothing. An unparseable date is unknown, not a crash."""
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("discarding unparseable datetime %r", value)
        return None


class LLMExtractor:
    """Extract entities and relations with Claude, validated against a schema.

    The client is injectable so tests can drive the mapping logic without
    network access.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        #: The raw response from the most recent call. The eval runner reads
        #: usage, the served model and stop_reason off this.
        self.last_response: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            # The SDK retries 429/5xx with backoff on its own (max_retries=2);
            # don't wrap this in a hand-rolled retry loop.
            self._client = anthropic.Anthropic()
        return self._client

    def extract(self, text: str, doc_id: str) -> Extraction:
        response = self._call(text)
        self.last_response = response
        raw = getattr(response, "parsed_output", None)
        if raw is None:
            raise ExtractionError(
                f"model returned no parsed output for {doc_id!r} "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r})"
            )
        return self._to_domain(raw, text=text, doc_id=doc_id)

    def _call(self, text: str) -> Any:
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            # The system prompt is identical on every document, so cache it:
            # stable content first, the varying document in the user turn.
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": text}],
            "output_format": _LLMExtraction,
            "thinking": {"type": "adaptive"},
        }
        if self._effort is not None:
            kwargs["output_config"] = {"effort": self._effort}

        try:
            return self.client.messages.parse(**kwargs)
        except anthropic.NotFoundError as exc:
            raise ExtractionError(f"unknown model {self._model!r}") from exc
        except anthropic.RateLimitError as exc:
            # Raised only after the SDK exhausted its own retries.
            raise ExtractionError("rate limited after SDK retries") from exc
        except anthropic.APIStatusError as exc:
            raise ExtractionError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ExtractionError("could not reach the API") from exc
        except ValidationError as exc:
            raise ExtractionError(f"response did not match the schema: {exc}") from exc
        except TypeError as exc:
            # The SDK raises a bare TypeError when no credential source
            # resolves. On its own that reads as a bug in this code.
            if "authentication" in str(exc).lower():
                raise ExtractionError(
                    "no API credentials found -- set ANTHROPIC_API_KEY or run `ant auth login`"
                ) from exc
            raise

    # -- mapping -------------------------------------------------------------

    def _to_domain(self, raw: _LLMExtraction, text: str, doc_id: str) -> Extraction:
        people: dict[str, Person] = {}
        for item in raw.people:
            if not item.name.strip():
                logger.warning("skipping person %r with empty name", item.id)
                continue
            people[item.id] = Person(name=item.name, gender=item.gender)

        places: dict[str, Place] = {
            item.id: Place(name=item.name, address=item.address, type=item.type)
            for item in raw.places
        }

        def person(ref: str, context: str) -> Person:
            try:
                return people[ref]
            except KeyError:
                raise ExtractionError(
                    f"{context} references undeclared person id {ref!r} "
                    f"(declared: {sorted(people)})"
                ) from None

        def place(ref: str, context: str) -> Place:
            try:
                return places[ref]
            except KeyError:
                raise ExtractionError(
                    f"{context} references undeclared place id {ref!r} "
                    f"(declared: {sorted(places)})"
                ) from None

        travels = [
            Travel(
                person=person(t.person_id, "travel"),
                place=place(t.place_id, "travel"),
                datetime=_parse_datetime(t.datetime),
            )
            for t in raw.travels
        ]

        activities = [
            ActivityEvent(
                person=person(a.person_id, f"activity {a.type!r}"),
                activity=Activity(
                    type=a.type,
                    datetime=_parse_datetime(a.datetime),
                    attributes=[
                        Attribute(name=attr.name, value=attr.value, unit=attr.unit)
                        for attr in a.attributes
                    ],
                ),
                place=(
                    place(a.place_id, f"activity {a.type!r}")
                    if a.place_id is not None
                    else None
                ),
            )
            for a in raw.activities
        ]

        # Only surface entities that ended up connected to something -- a
        # person or place the model declared but never used is noise, not a
        # fact about the document.
        used_places = [t.place for t in travels] + [
            a.place for a in activities if a.place is not None
        ]
        used_people = [t.person for t in travels] + [a.person for a in activities]

        return Extraction(
            doc_id=doc_id,
            text=text,
            people=_unique(used_people),
            places=_unique(used_places),
            travels=travels,
            activities=activities,
        )


def _unique(items: list) -> list:
    """De-duplicate by identity, preserving order."""
    out: list = []
    for item in items:
        if not any(existing is item for existing in out):
            out.append(item)
    return out
