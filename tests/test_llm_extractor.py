"""Tests for the LLM extractor's mapping and validation logic.

These never touch the network. `messages.parse` guarantees the response
matches the schema, so what is left to get wrong -- and what these cover --
is the mapping from the model's flat, id-referenced document onto the domain
model, and the referential-integrity checks around it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from graph_knowledge.extraction.llm import (
    ExtractionError,
    LLMExtractor,
    _LLMActivity,
    _LLMAttribute,
    _LLMExtraction,
    _LLMPerson,
    _LLMPlace,
    _LLMTravel,
)
from graph_knowledge.pipeline import Pipeline


class StubResponse:
    def __init__(self, parsed_output, stop_reason="end_turn"):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason


class StubMessages:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class StubClient:
    """Stands in for anthropic.Anthropic without any network access."""

    def __init__(self, parsed_output, stop_reason="end_turn"):
        self.messages = StubMessages(StubResponse(parsed_output, stop_reason))


def mary_payload() -> _LLMExtraction:
    """What the model should return for the Mary passage."""
    return _LLMExtraction(
        people=[_LLMPerson(id="mary", name="Mary", gender="Female")],
        places=[_LLMPlace(id="bank_1", name=None, address=None, type="Bank")],
        travels=[_LLMTravel(person_id="mary", place_id="bank_1", datetime=None)],
        activities=[
            _LLMActivity(
                person_id="mary",
                place_id="bank_1",
                type="withdraw money",
                datetime=None,
                attributes=[_LLMAttribute(name="amount", value=None, unit=None)],
            )
        ],
    )


def test_maps_model_output_onto_domain():
    extractor = LLMExtractor(client=StubClient(mary_payload()))

    extraction = extractor.extract("Mary went to the bank. She withdrew some money.", "mary")

    assert [p.name for p in extraction.people] == ["Mary"]
    assert extraction.people[0].gender == "Female"
    assert len(extraction.travels) == 1
    assert extraction.travels[0].place.type == "Bank"
    assert len(extraction.activities) == 1
    activity = extraction.activities[0].activity
    assert activity.type == "withdraw money"
    assert [(a.name, a.value) for a in activity.attributes] == [("amount", None)]


def test_shared_place_id_yields_one_place_object():
    """Both statements reference bank_1, so both must get the same Place."""
    extractor = LLMExtractor(client=StubClient(mary_payload()))

    extraction = extractor.extract("...", "mary")

    assert len(extraction.places) == 1
    assert extraction.travels[0].place is extraction.activities[0].place


def test_writes_the_expected_graph(store):
    """End to end through the store, with the same assertions as the rule-based path."""
    extractor = LLMExtractor(client=StubClient(mary_payload()))
    Pipeline(extractor, store).ingest("Mary went to the bank. She withdrew some money.", "mary")

    rows = store.activities_for("Mary")
    assert len(rows) == 1
    assert rows[0].activity_type == "withdraw money"
    assert rows[0].place_type == "Bank"
    assert rows[0].attribute_name == "amount"
    assert rows[0].attribute_value is None
    # Person + Place + Activity + Attribute, with the bank shared.
    assert store.node_count() == 4


def test_undeclared_person_id_fails_loudly():
    """A dangling reference is a bug to surface, not to write into the graph."""
    payload = mary_payload()
    payload.activities[0].person_id = "nobody"
    extractor = LLMExtractor(client=StubClient(payload))

    with pytest.raises(ExtractionError, match="undeclared person id 'nobody'"):
        extractor.extract("...", "mary")


def test_undeclared_place_id_fails_loudly():
    payload = mary_payload()
    payload.travels[0].place_id = "elsewhere"
    extractor = LLMExtractor(client=StubClient(payload))

    with pytest.raises(ExtractionError, match="undeclared place id 'elsewhere'"):
        extractor.extract("...", "mary")


def test_missing_parsed_output_fails_loudly():
    client = StubClient(None, stop_reason="max_tokens")
    with pytest.raises(ExtractionError, match="no parsed output"):
        LLMExtractor(client=client).extract("...", "mary")


def test_iso_datetime_is_parsed():
    payload = mary_payload()
    payload.travels[0].datetime = "2024-03-15T14:30:00Z"
    extraction = LLMExtractor(client=StubClient(payload)).extract("...", "mary")

    travel_time = extraction.travels[0].datetime
    assert travel_time == dt.datetime(2024, 3, 15, 14, 30, tzinfo=dt.timezone.utc)


def test_unparseable_datetime_becomes_unknown():
    """'last Tuesday' is unknown, not a reason to drop the whole document."""
    payload = mary_payload()
    payload.travels[0].datetime = "last Tuesday"
    extraction = LLMExtractor(client=StubClient(payload)).extract("...", "mary")

    assert extraction.travels[0].datetime is None


def test_unused_entities_are_dropped():
    """A declared-but-unreferenced entity is noise, not a fact."""
    payload = mary_payload()
    payload.people.append(_LLMPerson(id="ghost", name="Ghost", gender=None))
    payload.places.append(_LLMPlace(id="void", name=None, address=None, type="Void"))

    extraction = LLMExtractor(client=StubClient(payload)).extract("...", "mary")

    assert [p.name for p in extraction.people] == ["Mary"]
    assert [p.type for p in extraction.places] == ["Bank"]


def test_request_shape():
    """Schema, model and a cached system prompt on every call."""
    client = StubClient(mary_payload())
    LLMExtractor(client=client, model="claude-opus-5").extract("some text", "doc")

    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_format"] is _LLMExtraction
    assert call["thinking"] == {"type": "adaptive"}
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"] == [{"role": "user", "content": "some text"}]
    # Effort is left to the API default unless asked for.
    assert "output_config" not in call


def test_effort_is_forwarded_when_set():
    client = StubClient(mary_payload())
    LLMExtractor(client=client, effort="low").extract("some text", "doc")

    assert client.messages.calls[0]["output_config"] == {"effort": "low"}
