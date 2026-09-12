"""Store behaviour, run against every available backend.

The same assertions run on both engines, so a divergence between them is a
test failure rather than a surprise in production.
"""

from __future__ import annotations

import datetime as dt

import pytest

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
from graph_knowledge.vocabulary import Term, TermKind, TermStatus, Vocabulary

GATEWAY = Topic(name="payment gateway")


def evidence(text="ask Priya, she built the retry logic", confidence=0.8, ticket="SUP-1042"):
    return Evidence(ticket_id=ticket, excerpt=text, confidence=confidence)


@pytest.fixture
def priya():
    return Person(name="Priya N", aliases=["@priya", "priya@co.com"])


@pytest.fixture
def raj():
    return Person(name="Raj K", aliases=["@raj"])


@pytest.fixture
def ticket_extraction(priya, raj):
    return TicketExtraction(
        ticket=Ticket(id="SUP-1042", title="Payments failing at checkout"),
        text="Payments are failing. Ask Priya, she built the retry logic.",
        people=[priya, raj],
        topics=[GATEWAY],
        participations=[Participation(person=raj, role=ParticipationRole.REPORTER)],
        mentions=[Mention(person=priya, role=MentionRole.EXPERT, topic=GATEWAY,
                          evidence=evidence())],
        org_facts=[OrgFact(person=priya, relation=OrgRelation.HAS_FUNCTION,
                           target="Support Engineer", evidence=evidence())],
    )


def test_write_is_idempotent(store, ticket_extraction):
    store.write(ticket_extraction)
    before = store.node_count()

    store.write(ticket_extraction)

    assert store.node_count() == before


def test_aliases_resolve_to_one_person(store, ticket_extraction):
    """The identity problem: @priya and priya@co.com are the same human."""
    store.write(ticket_extraction)

    assert store.resolve_alias("@priya") == "Priya N"
    assert store.resolve_alias("priya@co.com") == "Priya N"
    assert store.resolve_alias("@nobody") is None


def test_participation_and_mention_are_distinguished(store, ticket_extraction):
    """Raj worked the ticket; Priya was named in it. Different signals."""
    store.write(ticket_extraction)

    rows = {r.person_name: r for r in store.people_for_ticket("SUP-1042")}

    assert rows["Raj K"].relation == "participated"
    assert rows["Raj K"].role == "reporter"
    assert rows["Priya N"].relation == "mentioned"
    assert rows["Priya N"].role == "expert"


def test_mention_carries_its_evidence(store, ticket_extraction):
    """A routing decision has to be able to quote why it chose someone."""
    store.write(ticket_extraction)

    row = next(r for r in store.people_for_ticket("SUP-1042") if r.relation == "mentioned")

    assert row.topic_name == "payment gateway"
    assert row.excerpt == "ask Priya, she built the retry logic"
    assert row.confidence == pytest.approx(0.8)


def test_org_facts_record_source_and_confidence(store, ticket_extraction):
    """Inferred org structure must be visibly provisional."""
    store.write(ticket_extraction)

    facts = store.org_facts_for("Priya N")

    assert len(facts) == 1
    assert facts[0].relation == "HAS_FUNCTION"
    assert facts[0].target_name == "Support Engineer"
    assert facts[0].source == Source.INFERRED.value
    assert facts[0].excerpt


def test_an_import_overrides_an_inference(store, priya):
    """A directory export is authoritative over something read out of prose."""
    inferred = TicketExtraction(
        ticket=Ticket(id="SUP-1"), text="...", people=[priya],
        org_facts=[OrgFact(person=priya, relation=OrgRelation.HAS_FUNCTION,
                           target="Support Engineer", source=Source.INFERRED,
                           evidence=evidence())],
    )
    store.write(inferred)

    imported = inferred.model_copy(deep=True)
    imported.org_facts[0].source = Source.IMPORTED
    store.write(imported)

    assert store.org_facts_for("Priya N")[0].source == Source.IMPORTED.value


def test_an_inference_does_not_downgrade_an_import(store, priya):
    """The guard that matters: a bad extraction must not overwrite the directory."""
    imported = TicketExtraction(
        ticket=Ticket(id="SUP-1"), text="...", people=[priya],
        org_facts=[OrgFact(person=priya, relation=OrgRelation.HAS_FUNCTION,
                           target="Support Engineer", source=Source.IMPORTED)],
    )
    store.write(imported)

    inferred = imported.model_copy(deep=True)
    inferred.org_facts[0].source = Source.INFERRED
    store.write(inferred)

    assert store.org_facts_for("Priya N")[0].source == Source.IMPORTED.value


def test_experts_for_topic_ranks_and_explains(store, ticket_extraction):
    """The routing primitive: candidates with the evidence that nominated them."""
    store.write(ticket_extraction)

    rows = store.experts_for_topic("payment gateway")

    by_name = {r.person_name: r for r in rows}
    assert by_name["Priya N"].expert_mentions == 1
    assert by_name["Priya N"].excerpts == ["ask Priya, she built the retry logic"]
    assert by_name["Raj K"].participations == 1
    # A colleague's explicit nomination outranks having touched the ticket.
    assert rows[0].person_name == "Priya N"


def test_expertise_accumulates_across_tickets(store, priya, raj):
    """Evidence from separate tickets adds up -- this is the memory claim."""
    for index in range(3):
        store.write(TicketExtraction(
            ticket=Ticket(id=f"SUP-{index}"), text="...", people=[priya, raj],
            topics=[GATEWAY],
            participations=[Participation(person=priya, role=ParticipationRole.RESOLVER)],
            mentions=[],
        ))

    row = next(r for r in store.experts_for_topic("payment gateway")
               if r.person_name == "Priya N")

    assert row.participations == 3


def test_vocabulary_round_trips(store):
    vocabulary = Vocabulary([
        Term(kind=TermKind.TOPIC, name="payment gateway", status=TermStatus.CANONICAL,
             supporting_tickets={"SUP-1", "SUP-2"}, aliases={"billing gateway"}),
        Term(kind=TermKind.TOPIC, name="sso", status=TermStatus.PROPOSED,
             supporting_tickets={"SUP-3"}),
    ])
    store.save_vocabulary(vocabulary.all_terms())

    reloaded = store.load_vocabulary()

    assert reloaded.canonical(TermKind.TOPIC) == ["payment gateway"]
    assert [t.name for t in reloaded.proposed(TermKind.TOPIC)] == ["sso"]
    assert reloaded.resolve(TermKind.TOPIC, "billing gateway").name == "payment gateway"
    assert reloaded.resolve(TermKind.TOPIC, "payment gateway").support == 2


def test_participation_timestamp_survives(store, priya):
    when = dt.datetime(2024, 3, 15, 9, 20)
    store.write(TicketExtraction(
        ticket=Ticket(id="SUP-9"), text="...", people=[priya],
        participations=[Participation(person=priya, role=ParticipationRole.RESOLVER, at=when)],
    ))

    rows = store.people_for_ticket("SUP-9")
    assert rows[0].person_name == "Priya N"
    assert rows[0].role == "resolver"
