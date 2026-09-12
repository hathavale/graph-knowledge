"""Score an extracted graph against a gold graph.

Both graphs are decomposed into atomic facts and compared as sets. Scoring
atomic facts rather than whole graphs means a single wrong attribute costs one
fact instead of zeroing the case, and it tells you *which* part regressed.

Fact kinds, each scored separately:

    person     (name)
    gender     (name, gender)             -- only when gender is asserted
    travel     (person, place)
    activity   (person, type, place)
    attribute  (person, type, attr, value)
    datetime   (person, type, iso)        -- only when a datetime is asserted

Gender and datetime facts exist only when asserted, so inventing one is a
false positive and missing one is a false negative. That is deliberate: "never
infer gender from a first name" and "don't resolve 'last Tuesday'" are
properties this eval has to be able to see.

Free-form strings (activity types, place types, attribute names) are the
brittle part of any structured-extraction grader: "flag outage" and "flag
payment outage" are both right. Each gold fact therefore carries an explicit
accept-list, visible in cases.jsonl and reviewable. `type_mismatches()`
reports near-misses so the accept-lists can be tuned from evidence rather than
guesswork.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable

__all__ = [
    "FACT_KINDS",
    "Metrics",
    "gold_facts",
    "predicted_facts",
    "score_case",
    "aggregate",
    "type_mismatches",
    "normalise",
]

FACT_KINDS = ("person", "gender", "travel", "activity", "attribute", "datetime")

UNKNOWN = "∅"  # explicit "asserted but unknown", distinct from absent
_DETERMINERS = ("the ", "a ", "an ", "some ", "their ", "his ", "her ")
_WS = re.compile(r"\s+")


def normalise(value: Any) -> str:
    """Lowercase, strip punctuation and a leading determiner.

    "The Bank" and "bank" are the same place; a grader that says otherwise is
    measuring formatting.
    """
    if value is None:
        return UNKNOWN
    text = _WS.sub(" ", str(value).strip().lower()).strip(" .,;:!?\"'")
    for determiner in _DETERMINERS:
        if text.startswith(determiner):
            text = text[len(determiner) :]
            break
    return text


def _number(value: Any) -> str:
    """Normalise numeric-looking values: 2,500 / 2500 / 2500.0 are one value."""
    text = normalise(value)
    stripped = text.replace(",", "")
    try:
        number = float(stripped)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def _place_variants(place: dict | None) -> tuple[str, ...]:
    """Every spelling of a gold place we are willing to accept."""
    if place is None:
        return ("",)
    variants = {normalise(v) for v in (place.get("name"), place.get("type")) if v}
    variants |= {normalise(a) for a in place.get("accept", ())}
    return tuple(sorted(variants)) or ("",)


def _place_key(place: Any) -> str:
    """How a *predicted* place is identified: its name, else its type."""
    if place is None:
        return ""
    name = getattr(place, "name", None)
    return normalise(name) if name else normalise(getattr(place, "type", None))


def _variants(primary: Any, accept: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({normalise(primary), *(normalise(a) for a in accept)}))


@dataclass
class GoldFact:
    kind: str
    variants: frozenset[tuple]
    label: str  # human-readable, for reports

    def matches(self, candidate: tuple) -> bool:
        return candidate in self.variants


def gold_facts(expected: dict) -> list[GoldFact]:
    facts: list[GoldFact] = []

    for person in expected.get("people", []):
        name = normalise(person["name"])
        facts.append(GoldFact("person", frozenset({("person", name)}), name))
        if person.get("gender"):
            gender = normalise(person["gender"])
            facts.append(
                GoldFact("gender", frozenset({("gender", name, gender)}), f"{name}={gender}")
            )

    for travel in expected.get("travels", []):
        person = normalise(travel["person"])
        places = _place_variants(travel["place"])
        facts.append(
            GoldFact(
                "travel",
                frozenset(("travel", person, place) for place in places),
                f"{person} -> {places[0]}",
            )
        )

    for activity in expected.get("activities", []):
        person = normalise(activity["person"])
        types = _variants(activity["type"], activity.get("type_accept", ()))
        places = _place_variants(activity.get("place"))
        facts.append(
            GoldFact(
                "activity",
                frozenset(
                    ("activity", person, t, p) for t, p in product(types, places)
                ),
                f"{person}: {types[0]} @ {places[0] or '-'}",
            )
        )

        if activity.get("datetime"):
            when = normalise(activity["datetime"])
            facts.append(
                GoldFact(
                    "datetime",
                    frozenset(("datetime", person, t, when) for t in types),
                    f"{person}: {types[0]} @ {when}",
                )
            )

        for attribute in activity.get("attributes", []):
            names = _variants(attribute["name"], attribute.get("name_accept", ()))
            value = _number(attribute.get("value"))
            facts.append(
                GoldFact(
                    "attribute",
                    frozenset(
                        ("attribute", person, t, n, value)
                        for t, n in product(types, names)
                    ),
                    f"{person}: {types[0]}.{names[0]}={value}",
                )
            )

    return facts


def predicted_facts(extraction: Any) -> list[tuple]:
    """Decompose an `Extraction` into the same fact tuples.

    Unit is deliberately not scored -- it is a detail that would make the
    attribute metric brittle. `type_mismatches()` surfaces it instead.
    """
    facts: list[tuple] = []

    for person in extraction.people:
        name = normalise(person.name)
        facts.append(("person", name))
        if person.gender:
            facts.append(("gender", name, normalise(person.gender)))

    for travel in extraction.travels:
        facts.append(("travel", normalise(travel.person.name), _place_key(travel.place)))

    for event in extraction.activities:
        person = normalise(event.person.name)
        atype = normalise(event.activity.type)
        facts.append(("activity", person, atype, _place_key(event.place)))

        if event.activity.datetime is not None:
            facts.append(
                ("datetime", person, atype, normalise(event.activity.datetime.isoformat()))
            )

        for attribute in event.activity.attributes:
            facts.append(
                ("attribute", person, atype, normalise(attribute.name), _number(attribute.value))
            )

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
        # Nothing predicted and nothing expected is perfect precision, not zero.
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.missed + other.missed,
            self.spurious + other.spurious,
        )

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def score_case(expected: dict, extraction: Any) -> dict[str, Metrics]:
    """Greedy set matching, one predicted fact consumed per gold fact."""
    gold = gold_facts(expected)
    predicted = predicted_facts(extraction)
    unconsumed = list(predicted)
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
        kind = leftover[0]
        per_kind[kind].fp += 1
        per_kind[kind].spurious.append(leftover)

    return per_kind


def aggregate(per_case: Iterable[dict[str, Metrics]]) -> dict[str, dict]:
    """Micro-average over cases: sum the counts, then compute the rates.

    Micro rather than macro so every fact carries equal weight; a case with
    one fact should not outvote a case with eight.
    """
    totals = {kind: Metrics() for kind in FACT_KINDS}
    for case in per_case:
        for kind, metrics in case.items():
            totals[kind] = totals[kind] + metrics

    out = {kind: metrics.as_dict() for kind, metrics in totals.items()}
    overall = Metrics()
    for metrics in totals.values():
        overall = overall + metrics
    out["overall"] = overall.as_dict()
    return out


def type_mismatches(expected: dict, extraction: Any) -> list[dict]:
    """Near-misses: right person and place, wrong type string.

    These are the candidates for an accept-list entry -- a way to tune the
    grader from evidence instead of guessing at synonyms up front.
    """
    gold = [f for f in gold_facts(expected) if f.kind == "activity"]
    predicted = [f for f in predicted_facts(extraction) if f[0] == "activity"]
    out = []
    for fact in gold:
        if any(fact.matches(p) for p in predicted):
            continue
        people = {v[1] for v in fact.variants}
        places = {v[3] for v in fact.variants}
        for candidate in predicted:
            if candidate[1] in people and candidate[3] in places:
                out.append(
                    {"gold": fact.label, "predicted_type": candidate[2],
                     "accepted_types": sorted({v[2] for v in fact.variants})}
                )
    return out
