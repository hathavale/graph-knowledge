"""Tests for the eval harness itself.

An eval that silently breaks reports confident numbers pointing the wrong way,
so the two bounds are enforced here rather than checked by hand: the oracle
(gold replayed) must score 1.0, and the null extractor must score 0. If the
oracle drops, the harness or grader is broken; if the null scores above zero,
the grader is too lenient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from grader import aggregate, normalise, score_case  # noqa: E402
from run_eval import NullExtractor, OracleExtractor, load_cases, make_split  # noqa: E402

from graph_knowledge.models import Activity, ActivityEvent, Attribute, Extraction, Person, Place


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def _score_all(cases, extractor):
    return aggregate(
        score_case(c["expected"], extractor.extract(c["text"], c["id"])) for c in cases
    )


def test_oracle_scores_perfectly(cases):
    """Gold replayed through the grader must come back 1.0 on every fact kind."""
    totals = _score_all(cases, OracleExtractor(cases))

    assert totals["overall"]["f1"] == 1.0
    assert totals["overall"]["fp"] == 0
    assert totals["overall"]["fn"] == 0


def test_null_baseline_scores_zero(cases):
    """Extracting nothing must not score above zero, or the grader is lenient."""
    totals = _score_all(cases, NullExtractor())

    assert totals["overall"]["f1"] == 0.0
    assert totals["overall"]["tp"] == 0


def test_negative_cases_reward_extracting_nothing(cases):
    """On a passage with no facts, the right answer is an empty graph."""
    negatives = [c for c in cases if c["tags"][0] == "negative"]
    assert negatives, "the set must contain negative cases"

    totals = _score_all(negatives, NullExtractor())

    assert totals["overall"]["f1"] == 1.0


def test_gold_has_no_facts_on_negative_cases(cases):
    for case in cases:
        if case["tags"][0] == "negative":
            assert case["expected"]["people"] == []
            assert case["expected"]["activities"] == []


# --- grader behaviour -------------------------------------------------------


def _extraction(person="Mary", gender=None, activity=None, place=None, attributes=(), when=None):
    subject = Person(name=person, gender=gender)
    events = []
    if activity:
        events.append(
            ActivityEvent(
                person=subject,
                activity=Activity(
                    type=activity,
                    datetime=when,
                    attributes=[Attribute(name=n, value=v) for n, v in attributes],
                ),
                place=place,
            )
        )
    return Extraction(doc_id="t", text="t", people=[subject], activities=events)


GOLD = {
    "people": [{"name": "Mary", "gender": "Female"}],
    "travels": [],
    "activities": [
        {
            "person": "Mary",
            "type": "withdraw money",
            "type_accept": ["withdraw cash"],
            "place": {"name": None, "type": "Bank", "accept": ["bank"]},
            "attributes": [{"name": "amount", "value": None, "unit": None, "name_accept": []}],
        }
    ],
}


def test_normalisation_ignores_case_and_determiner():
    assert normalise("The Bank") == normalise("bank") == "bank"


def test_accept_list_admits_a_synonym():
    """'withdraw cash' is listed as acceptable, so it must not be a miss."""
    got = score_case(
        GOLD,
        _extraction("Mary", "Female", "withdraw cash", Place(type="Bank"), [("amount", None)]),
    )
    assert got["activity"].fn == 0
    assert got["activity"].tp == 1


def test_unlisted_paraphrase_is_a_miss():
    """The accept-list is explicit; anything outside it counts against the score."""
    got = score_case(
        GOLD, _extraction("Mary", "Female", "take out money", Place(type="Bank"), [("amount", None)])
    )
    assert got["activity"].fn == 1
    assert got["activity"].fp == 1


def test_invented_gender_is_a_false_positive():
    """Gender asserted where gold asserts none must cost precision."""
    gold = {"people": [{"name": "Alex", "gender": None}], "travels": [], "activities": []}

    got = score_case(gold, _extraction("Alex", "Male"))

    assert got["gender"].fp == 1
    assert got["person"].tp == 1


def test_missing_gender_is_a_false_negative():
    gold = {"people": [{"name": "Mary", "gender": "Female"}], "travels": [], "activities": []}

    got = score_case(gold, _extraction("Mary", None))

    assert got["gender"].fn == 1


def test_invented_datetime_is_a_false_positive():
    """'last Tuesday' resolved into a date must be penalised, not rewarded."""
    import datetime as dt

    gold = {
        "people": [{"name": "Mary", "gender": None}],
        "travels": [],
        "activities": [{"person": "Mary", "type": "deploy hotfix", "type_accept": [],
                        "place": None, "attributes": []}],
    }

    got = score_case(gold, _extraction("Mary", None, "deploy hotfix", when=dt.datetime(2024, 3, 15)))

    assert got["datetime"].fp == 1
    assert got["activity"].tp == 1


def test_unknown_attribute_value_differs_from_a_stated_one():
    """An invented amount must not satisfy a gold attribute whose value is unknown."""
    got = score_case(
        GOLD, _extraction("Mary", "Female", "withdraw money", Place(type="Bank"), [("amount", "500")])
    )
    assert got["attribute"].fn == 1
    assert got["attribute"].fp == 1


def test_numeric_values_normalise():
    gold = {
        "people": [{"name": "Mary", "gender": None}], "travels": [],
        "activities": [{"person": "Mary", "type": "transfer money", "type_accept": [], "place": None,
                        "attributes": [{"name": "amount", "value": "2500", "unit": None, "name_accept": []}]}],
    }

    got = score_case(gold, _extraction("Mary", None, "transfer money", attributes=[("amount", "2,500")]))

    assert got["attribute"].tp == 1


# --- split ------------------------------------------------------------------


def test_split_is_deterministic_and_proportioned(cases):
    first, second = make_split(cases), make_split(cases)
    assert first == second
    assert sum(1 for v in first.values() if v == "train") == 28
    assert sum(1 for v in first.values() if v == "test") == 12


def test_split_covers_every_tag_on_both_sides(cases):
    """A held-out slice missing a whole category cannot confirm anything about it."""
    split = make_split(cases)
    for slice_name in ("train", "test"):
        tags = {c["tags"][0] for c in cases if split[c["id"]] == slice_name}
        assert tags == {c["tags"][0] for c in cases}


def test_every_case_is_assigned_exactly_once(cases):
    split = make_split(cases)
    assert set(split) == {c["id"] for c in cases}
