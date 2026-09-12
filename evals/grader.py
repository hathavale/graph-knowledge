"""Score an extracted ticket graph against a gold graph.

Atomic facts compared as sets, scored per kind, so a regression is
attributable rather than showing up as one blended number.

    person        (name)
    alias         (name, alias)
    topic         (topic)
    participation (person, role)
    mention       (person, role, topic)
    org           (person, relation, target)

`mention` and `org` are the kinds that matter most for routing: a mention with
the wrong role sends the notification to someone for the wrong reason, and an
org fact invented from prose is how a tool ends up escalating to the wrong
manager. Both are scored strictly, and `org` recall is deliberately *not*
something to maximise -- several cases assert that no org fact should be
produced at all, so inventing one costs precision.

Free-form strings (topic names, function and team names) carry explicit
accept-lists in cases.jsonl, visible and reviewable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable

__all__ = [
    "FACT_KINDS", "Metrics", "gold_facts", "predicted_facts",
    "score_case", "aggregate", "normalise",
]

FACT_KINDS = ("person", "alias", "topic", "participation", "mention", "org")

_WS = re.compile(r"\s+")


def normalise(value: Any) -> str:
    if value is None:
        return ""
    return _WS.sub(" ", str(value).strip().lower()).strip(" .,;:!?\"'")


def _variants(primary: Any, accept: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({normalise(primary), *(normalise(a) for a in accept)}))


@dataclass
class GoldFact:
    kind: str
    variants: frozenset[tuple]
    label: str

    def matches(self, candidate: tuple) -> bool:
        return candidate in self.variants


def gold_facts(expected: dict) -> list[GoldFact]:
    facts: list[GoldFact] = []
    topic_variants: dict[str, tuple[str, ...]] = {}

    for topic in expected.get("topics", []):
        names = _variants(topic["name"], topic.get("accept", ()))
        topic_variants[normalise(topic["name"])] = names
        facts.append(GoldFact("topic", frozenset(("topic", n) for n in names), names[0]))

    for person in expected.get("people", []):
        name = normalise(person["name"])
        facts.append(GoldFact("person", frozenset({("person", name)}), name))
        for alias in person.get("aliases", []):
            facts.append(GoldFact(
                "alias", frozenset({("alias", name, normalise(alias))}), f"{name}~{alias}"
            ))

    for item in expected.get("participations", []):
        person, role = normalise(item["person"]), normalise(item["role"])
        facts.append(GoldFact(
            "participation", frozenset({("participation", person, role)}), f"{person}:{role}"
        ))

    for item in expected.get("mentions", []):
        person, role = normalise(item["person"]), normalise(item["role"])
        topics = topic_variants.get(normalise(item.get("topic")), (normalise(item.get("topic")),))
        facts.append(GoldFact(
            "mention",
            frozenset(("mention", person, role, t) for t in topics),
            f"{person}:{role}@{topics[0] or '-'}",
        ))

    for item in expected.get("org_facts", []):
        person = normalise(item["person"])
        relation = normalise(item["relation"])
        targets = _variants(item["target"], item.get("accept", ()))
        facts.append(GoldFact(
            "org",
            frozenset(("org", person, relation, t) for t in targets),
            f"{person} {relation} {targets[0]}",
        ))

    return facts


def predicted_facts(extraction: Any) -> list[tuple]:
    facts: list[tuple] = []

    for person in extraction.people:
        name = normalise(person.name)
        facts.append(("person", name))
        for alias in person.aliases:
            facts.append(("alias", name, normalise(alias)))

    for topic in extraction.topics:
        facts.append(("topic", normalise(topic.name)))

    for item in extraction.participations:
        facts.append(("participation", normalise(item.person.name), normalise(item.role.value)))

    for item in extraction.mentions:
        facts.append((
            "mention", normalise(item.person.name), normalise(item.role.value),
            normalise(item.topic.name) if item.topic else "",
        ))

    for item in extraction.org_facts:
        facts.append((
            "org", normalise(item.person.name),
            normalise(item.relation.value), normalise(item.target),
        ))

    return facts


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    missed: list[str] = field(default_factory=list)
    spurious: list[tuple] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn,
                       self.missed + other.missed, self.spurious + other.spurious)

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4)}


def score_case(expected: dict, extraction: Any) -> dict[str, Metrics]:
    gold = gold_facts(expected)
    unconsumed = list(predicted_facts(extraction))
    per_kind = {kind: Metrics() for kind in FACT_KINDS}

    for fact in gold:
        hit = next((c for c in unconsumed if fact.matches(c)), None)
        if hit is not None:
            unconsumed.remove(hit)
            per_kind[fact.kind].tp += 1
        else:
            per_kind[fact.kind].fn += 1
            per_kind[fact.kind].missed.append(fact.label)

    for leftover in unconsumed:
        per_kind[leftover[0]].fp += 1
        per_kind[leftover[0]].spurious.append(leftover)

    return per_kind


def aggregate(per_case: Iterable[dict[str, Metrics]]) -> dict[str, dict]:
    totals = {kind: Metrics() for kind in FACT_KINDS}
    for case in per_case:
        for kind, metrics in case.items():
            totals[kind] = totals[kind] + metrics

    out = {kind: m.as_dict() for kind, m in totals.items()}
    overall = Metrics()
    for metrics in totals.values():
        overall = overall + metrics
    out["overall"] = overall.as_dict()
    return out
