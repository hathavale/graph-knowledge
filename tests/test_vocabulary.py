"""The curation gate.

The requirement is a model that evolves with LLM input. These tests pin the
difference between evolving and drifting.
"""

from __future__ import annotations

from graph_knowledge.vocabulary import TermKind, TermStatus, Vocabulary


def test_term_starts_proposed_not_canonical():
    """A term the model invented is not trusted on first sight."""
    v = Vocabulary(threshold=3)

    term = v.observe(TermKind.TOPIC, "payment gateway", "SUP-1")

    assert term.status is TermStatus.PROPOSED
    assert v.canonical(TermKind.TOPIC) == []


def test_support_counts_distinct_tickets_not_mentions():
    """One ticket repeating a term five times is still one piece of evidence."""
    v = Vocabulary(threshold=3)

    for _ in range(5):
        v.observe(TermKind.TOPIC, "payment gateway", "SUP-1")

    assert v.resolve(TermKind.TOPIC, "payment gateway").support == 1
    assert v.canonical(TermKind.TOPIC) == []


def test_promotes_once_enough_tickets_agree():
    v = Vocabulary(threshold=3)

    for ticket in ("SUP-1", "SUP-2", "SUP-3"):
        v.observe(TermKind.TOPIC, "payment gateway", ticket)

    assert v.canonical(TermKind.TOPIC) == ["payment gateway"]


def test_resolution_is_case_and_punctuation_insensitive():
    """`Payment Gateway` must not become a second topic."""
    v = Vocabulary(threshold=2)
    v.observe(TermKind.TOPIC, "payment gateway", "SUP-1")
    v.observe(TermKind.TOPIC, "Payment  Gateway", "SUP-2")

    assert len(v.all_terms()) == 1
    assert v.canonical(TermKind.TOPIC) == ["payment gateway"]


def test_alias_merges_a_duplicate_and_keeps_its_evidence():
    """The repair tool: two names for one concept, folded without losing support."""
    v = Vocabulary(threshold=10)
    v.observe(TermKind.TOPIC, "payment gateway", "SUP-1")
    v.observe(TermKind.TOPIC, "billing gateway", "SUP-2")

    v.alias(TermKind.TOPIC, "payment gateway", "billing gateway")

    assert len(v.all_terms()) == 1
    term = v.resolve(TermKind.TOPIC, "billing gateway")
    assert term.name == "payment gateway"
    assert term.support == 2


def test_rejected_term_is_not_resurrected_by_repetition():
    """A curation decision must not be undone by weight of evidence."""
    v = Vocabulary(threshold=2)
    v.observe(TermKind.TOPIC, "misc", "SUP-1")
    v.reject(TermKind.TOPIC, "misc")

    for ticket in ("SUP-2", "SUP-3", "SUP-4"):
        v.observe(TermKind.TOPIC, "misc", ticket)

    assert v.resolve(TermKind.TOPIC, "misc").status is TermStatus.REJECTED
    assert v.canonical(TermKind.TOPIC) == []


def test_kinds_are_separate_namespaces():
    v = Vocabulary(threshold=1)
    v.observe(TermKind.TOPIC, "billing", "SUP-1")
    v.observe(TermKind.TEAM, "billing", "SUP-1")

    assert v.canonical(TermKind.TOPIC) == ["billing"]
    assert v.canonical(TermKind.TEAM) == ["billing"]
    assert len(v.all_terms()) == 2


def test_proposed_are_ranked_by_support():
    """The curation queue shows the best-evidenced proposals first."""
    v = Vocabulary(threshold=99)
    v.observe(TermKind.TOPIC, "rare", "SUP-1")
    for ticket in ("SUP-1", "SUP-2", "SUP-3"):
        v.observe(TermKind.TOPIC, "common", ticket)

    assert [t.name for t in v.proposed(TermKind.TOPIC)] == ["common", "rare"]
