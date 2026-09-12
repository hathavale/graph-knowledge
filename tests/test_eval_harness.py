"""Tests for the eval harness itself.

The two bounds are enforced here rather than checked by hand: the oracle
(gold replayed) must score 1.0, and the null extractor must score 0. If the
oracle drops, the harness or grader is broken; if null scores above zero, the
grader is too lenient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from grader import aggregate, normalise, score_case  # noqa: E402
from run_eval import NullExtractor, OracleExtractor, load_cases, make_split  # noqa: E402

from graph_knowledge.models import (
    Mention, MentionRole, OrgFact, OrgRelation, Participation, ParticipationRole,
    Person, Ticket, TicketExtraction, Topic,
)


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def _score_all(cases, extractor):
    return aggregate(
        score_case(c["expected"], extractor.extract(c["text"], Ticket(id=c["id"])))
        for c in cases
    )


def test_oracle_scores_perfectly(cases):
    totals = _score_all(cases, OracleExtractor(cases))

    assert totals["overall"]["f1"] == 1.0
    assert totals["overall"]["fp"] == 0
    assert totals["overall"]["fn"] == 0


def test_null_baseline_scores_zero(cases):
    totals = _score_all(cases, NullExtractor())

    assert totals["overall"]["f1"] == 0.0
    assert totals["overall"]["tp"] == 0


def test_negative_cases_reward_extracting_nothing(cases):
    negatives = [c for c in cases if c["tags"][0] == "negative"]
    assert negatives

    assert _score_all(negatives, NullExtractor())["overall"]["f1"] == 1.0


def test_most_cases_assert_no_org_fact(cases):
    """Org structure is the least reliable inference, so the set pushes against it."""
    without = [c for c in cases if not c["expected"]["org_facts"]]

    assert len(without) >= len(cases) * 0.7


# --- grader behaviour -------------------------------------------------------


def _extraction(people=(), topics=(), participations=(), mentions=(), org_facts=()):
    return TicketExtraction(
        ticket=Ticket(id="t"), text="t", people=list(people), topics=list(topics),
        participations=list(participations), mentions=list(mentions),
        org_facts=list(org_facts),
    )


def test_normalisation_ignores_case_and_spacing():
    assert normalise("  Payment  Gateway ") == normalise("payment gateway")


def test_accept_list_admits_a_synonym():
    gold = {"people": [], "topics": [{"name": "payment gateway", "accept": ["payments"]}],
            "participations": [], "mentions": [], "org_facts": []}

    got = score_case(gold, _extraction(topics=[Topic(name="payments")]))

    assert got["topic"].tp == 1
    assert got["topic"].fn == 0


def test_invented_org_fact_is_a_false_positive():
    """The failure that messages the wrong manager."""
    gold = {"people": [{"name": "Priya"}], "topics": [], "participations": [],
            "mentions": [], "org_facts": []}
    priya = Person(name="Priya")

    got = score_case(gold, _extraction(
        people=[priya],
        org_facts=[OrgFact(person=priya, relation=OrgRelation.REPORTS_TO, target="Meera")],
    ))

    assert got["org"].fp == 1
    assert got["person"].tp == 1


def test_wrong_mention_role_is_both_a_miss_and_a_false_positive():
    """Notifying the escalation contact as an expert is not a partial success."""
    gold = {"people": [{"name": "Meera"}], "topics": [{"name": "sso"}],
            "participations": [],
            "mentions": [{"person": "Meera", "role": "escalation", "topic": "sso"}],
            "org_facts": []}
    meera = Person(name="Meera")

    got = score_case(gold, _extraction(
        people=[meera], topics=[Topic(name="sso")],
        mentions=[Mention(person=meera, role=MentionRole.EXPERT, topic=Topic(name="sso"))],
    ))

    assert got["mention"].fn == 1
    assert got["mention"].fp == 1


def test_missing_alias_is_a_false_negative():
    """A dropped handle means a later ticket creates a duplicate person."""
    gold = {"people": [{"name": "Priya", "aliases": ["@priya"]}], "topics": [],
            "participations": [], "mentions": [], "org_facts": []}

    got = score_case(gold, _extraction(people=[Person(name="Priya")]))

    assert got["alias"].fn == 1
    assert got["person"].tp == 1


def test_participation_role_is_scored():
    gold = {"people": [{"name": "Raj"}], "topics": [],
            "participations": [{"person": "Raj", "role": "reporter"}],
            "mentions": [], "org_facts": []}
    raj = Person(name="Raj")

    got = score_case(gold, _extraction(
        people=[raj],
        participations=[Participation(person=raj, role=ParticipationRole.RESOLVER)],
    ))

    assert got["participation"].fn == 1
    assert got["participation"].fp == 1


# --- split ------------------------------------------------------------------


def test_split_is_deterministic(cases):
    assert make_split(cases) == make_split(cases)


def test_split_covers_every_tag_on_both_sides(cases):
    split = make_split(cases)
    for slice_name in ("train", "test"):
        tags = {c["tags"][0] for c in cases if split[c["id"]] == slice_name}
        assert tags == {c["tags"][0] for c in cases}


def test_every_case_assigned_once(cases):
    assert set(make_split(cases)) == {c["id"] for c in cases}
