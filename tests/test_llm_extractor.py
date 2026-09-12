"""Extractor mapping and validation. No network: the client is a stub.

`messages.parse` guarantees the response matches the schema, so what remains
to get wrong -- and what these cover -- is the mapping onto the domain model,
referential integrity, and the vocabulary constraint.
"""

from __future__ import annotations

import pytest

from graph_knowledge.extraction.llm import (
    ExtractionError,
    LLMExtractor,
    _LLMMention,
    _LLMOrgFact,
    _LLMParticipation,
    _LLMPerson,
    _LLMTicketExtraction,
    _LLMTopic,
)
from graph_knowledge.models import MentionRole, ParticipationRole, Source, Ticket
from graph_knowledge.pipeline import Pipeline
from graph_knowledge.vocabulary import Term, TermKind, TermStatus, Vocabulary


class StubMessages:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class StubResponse:
    def __init__(self, parsed_output, stop_reason="end_turn"):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason
        self.model = "claude-opus-5"
        self.usage = None


class StubClient:
    def __init__(self, parsed_output, stop_reason="end_turn"):
        self.messages = StubMessages(StubResponse(parsed_output, stop_reason))


def payload(topic_name="payment gateway", is_new=False) -> _LLMTicketExtraction:
    return _LLMTicketExtraction(
        people=[
            _LLMPerson(id="priya", name="Priya N", aliases=["@priya", "priya@co.com"]),
            _LLMPerson(id="raj", name="Raj K", aliases=["@raj"]),
        ],
        topics=[_LLMTopic(id="gw", name=topic_name, is_new=is_new)],
        participations=[_LLMParticipation(person_id="raj", role="reporter", at=None)],
        mentions=[_LLMMention(person_id="priya", role="expert", topic_id="gw",
                              excerpt="ask Priya, she built the retry logic",
                              confidence=0.8)],
        org_facts=[_LLMOrgFact(person_id="priya", relation="HAS_FUNCTION",
                               target="Support Engineer",
                               excerpt="Priya, our support engineer", confidence=0.6)],
    )


def canonical_vocabulary() -> Vocabulary:
    return Vocabulary([
        Term(kind=TermKind.TOPIC, name="payment gateway", status=TermStatus.CANONICAL,
             aliases={"billing gateway"}),
    ])


def test_maps_output_onto_the_domain_model():
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(payload()))

    result = extractor.extract("...", Ticket(id="SUP-1042"))

    assert {p.name for p in result.people} == {"Priya N", "Raj K"}
    assert result.participations[0].role is ParticipationRole.REPORTER
    assert result.mentions[0].role is MentionRole.EXPERT
    assert result.mentions[0].topic.name == "payment gateway"
    assert result.org_facts[0].source is Source.INFERRED


def test_evidence_is_attached_to_every_inferred_fact():
    """Without provenance a notification cannot explain itself."""
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(payload()))

    result = extractor.extract("...", "SUP-1042")

    assert result.mentions[0].evidence.excerpt == "ask Priya, she built the retry logic"
    assert result.mentions[0].evidence.ticket_id == "SUP-1042"
    assert result.org_facts[0].evidence.confidence == pytest.approx(0.6)


def test_aliases_are_carried_through():
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(payload()))

    result = extractor.extract("...", "SUP-1042")

    priya = next(p for p in result.people if p.name == "Priya N")
    assert set(priya.aliases) == {"@priya", "priya@co.com"}


def test_a_known_alias_folds_onto_its_canonical_topic():
    """'billing gateway' must not enter the graph as a second topic."""
    extractor = LLMExtractor(
        vocabulary=canonical_vocabulary(),
        client=StubClient(payload(topic_name="billing gateway", is_new=True)),
    )

    result = extractor.extract("...", "SUP-1042")

    assert [t.name for t in result.topics] == ["payment gateway"]
    assert result.proposed_terms == []


def test_a_genuinely_new_topic_is_proposed_not_adopted():
    extractor = LLMExtractor(
        vocabulary=canonical_vocabulary(),
        client=StubClient(payload(topic_name="sso login loop", is_new=True)),
    )

    result = extractor.extract("...", "SUP-1042")

    assert result.proposed_terms == ["sso login loop"]


def test_canonical_terms_appear_in_the_prompt():
    """The vocabulary constrains extraction by being in the system prompt."""
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(payload()))

    prompt = extractor.system_prompt()

    assert "- payment gateway" in prompt


def test_undeclared_person_id_fails_loudly():
    broken = payload()
    broken.mentions[0].person_id = "nobody"
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(broken))

    with pytest.raises(ExtractionError, match="undeclared person id 'nobody'"):
        extractor.extract("...", "SUP-1042")


def test_undeclared_topic_id_fails_loudly():
    broken = payload()
    broken.mentions[0].topic_id = "ghost"
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(broken))

    with pytest.raises(ExtractionError, match="undeclared topic id 'ghost'"):
        extractor.extract("...", "SUP-1042")


def test_unknown_role_degrades_instead_of_failing():
    """A bad enum loses precision; it should not discard the whole ticket."""
    odd = payload()
    odd.mentions[0].role = "wizard"
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(odd))

    result = extractor.extract("...", "SUP-1042")

    assert result.mentions[0].role is MentionRole.UNSPECIFIED


def test_unknown_org_relation_is_dropped():
    """An unmappable relation has nowhere to go; better dropped than guessed."""
    odd = payload()
    odd.org_facts[0].relation = "SITS_NEAR"
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(odd))

    assert extractor.extract("...", "SUP-1042").org_facts == []


def test_missing_parsed_output_fails_loudly():
    extractor = LLMExtractor(client=StubClient(None, stop_reason="max_tokens"))

    with pytest.raises(ExtractionError, match="no parsed output"):
        extractor.extract("...", "SUP-1042")


def test_relative_datetime_is_not_invented():
    odd = payload()
    odd.participations[0].at = "yesterday"
    extractor = LLMExtractor(vocabulary=canonical_vocabulary(), client=StubClient(odd))

    assert extractor.extract("...", "SUP-1042").participations[0].at is None


def test_request_shape():
    client = StubClient(payload())
    LLMExtractor(vocabulary=canonical_vocabulary(), client=client).extract("body", "SUP-1")

    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_format"] is _LLMTicketExtraction
    assert call["thinking"] == {"type": "adaptive"}
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"] == [{"role": "user", "content": "body"}]


# --- pipeline integration ---------------------------------------------------


def test_pipeline_writes_and_grows_the_vocabulary(store):
    """One pass: extract, observe terms, persist, and expose the routing view."""
    vocabulary = Vocabulary(threshold=2)
    extractor = LLMExtractor(vocabulary=vocabulary, client=StubClient(payload("sso", is_new=True)))
    pipeline = Pipeline(extractor, store, vocabulary)

    pipeline.ingest("...", "SUP-1")
    assert vocabulary.canonical(TermKind.TOPIC) == []      # one ticket is not enough

    extractor._client = StubClient(payload("sso", is_new=True))
    pipeline.ingest("...", "SUP-2")

    assert vocabulary.canonical(TermKind.TOPIC) == ["sso"]
    assert store.load_vocabulary().canonical(TermKind.TOPIC) == ["sso"]
    assert [r.person_name for r in store.experts_for_topic("sso")][0] == "Priya N"
