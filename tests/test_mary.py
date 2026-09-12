"""The worked example.

    "Mary went to the bank. She withdrew some money."

must become:

    (:Person {name:"Mary", gender:"Female"})-[:TRAVEL]->(:Place {type:"Bank"})
    (:Person {name:"Mary", gender:"Female"})-[:PERFORMED]->
        (:Activity {type:"withdraw money"})-[:AT]->(:Place {type:"Bank"})
    (:Activity)-[:HAS_ATTRIBUTE]->(:Attribute {name:"amount", value:null})
"""

from __future__ import annotations

import pytest

from graph_knowledge.extraction import RuleBasedExtractor
from graph_knowledge.pipeline import Pipeline

MARY = "Mary went to the bank. She withdrew some money."


@pytest.fixture
def pipeline(store):
    return Pipeline(RuleBasedExtractor(), store)


def test_extracts_person_with_gender_from_pronoun():
    """'She' in sentence two is what tells us Mary's gender."""
    extraction = RuleBasedExtractor().extract(MARY, doc_id="mary")

    assert len(extraction.people) == 1
    mary = extraction.people[0]
    assert mary.name == "Mary"
    assert mary.gender == "Female"


def test_travel_edge(pipeline, store):
    pipeline.ingest(MARY, doc_id="mary")

    travels = store.travels_for("Mary")
    assert len(travels) == 1
    travel = travels[0]
    assert travel.person_name == "Mary"
    assert travel.person_gender == "Female"
    assert travel.place_type == "Bank"
    # Unknowns stay unknown rather than being invented.
    assert travel.place_name is None
    assert travel.travel_datetime is None


def test_activity_with_unknown_attribute_value(pipeline, store):
    pipeline.ingest(MARY, doc_id="mary")

    rows = store.activities_for("Mary")
    assert len(rows) == 1
    row = rows[0]
    assert row.person_name == "Mary"
    assert row.person_gender == "Female"
    assert row.activity_type == "withdraw money"
    assert row.activity_datetime is None
    assert row.place_type == "Bank"
    # "some money": we know an amount exists, we don't know what it was.
    assert row.attribute_name == "amount"
    assert row.attribute_value is None


def test_both_statements_share_one_place_node(pipeline, store):
    """The travel and the activity must point at the *same* bank."""
    pipeline.ingest(MARY, doc_id="mary")

    # Person + Place + Activity + Attribute == 4. A fifth node would mean the
    # anonymous bank was duplicated across the two sentences.
    assert store.node_count() == 4


def test_ingest_is_idempotent(pipeline, store):
    """Re-reading a document must not duplicate anything."""
    pipeline.ingest(MARY, doc_id="mary")
    before = store.node_count()

    pipeline.ingest(MARY, doc_id="mary")

    assert store.node_count() == before
    assert len(store.travels_for("Mary")) == 1
    assert len(store.activities_for("Mary")) == 1


def test_unnamed_places_do_not_merge_across_documents(pipeline, store):
    """Two documents' anonymous banks are different banks."""
    pipeline.ingest(MARY, doc_id="mary")
    pipeline.ingest("John went to the bank. He withdrew some money.", doc_id="john")

    mary_place = store.travels_for("Mary")[0]
    john_place = store.travels_for("John")[0]
    assert mary_place.place_type == john_place.place_type == "Bank"

    # 2 people + 2 places + 2 activities + 2 attributes
    assert store.node_count() == 8


def test_stated_amount_is_captured(pipeline, store):
    """When the text supplies the value, the attribute is filled in."""
    pipeline.ingest("Alice went to the bank. She withdrew 500 dollars.", doc_id="alice")

    rows = store.activities_for("Alice")
    assert len(rows) == 1
    assert rows[0].attribute_name == "amount"
    assert rows[0].attribute_value == "500"
