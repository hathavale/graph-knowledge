"""A deterministic baseline extractor.

Deliberately small and explainable: it resolves pronouns to the most recent
person, infers gender from pronouns, and recognises travel and activity
clauses from seed lexicons. It exists so the pipeline, the store and the tests
are exercisable offline with no API key and no model download.

It is a baseline, not the destination. Real coverage of "ideas, places, people
and gender" needs statistical coreference (fastcoref / maverick-coref) and an
LLM constrained to the `Extraction` schema. Both plug in behind `Extractor`
without touching anything else.
"""

from __future__ import annotations

import re

from graph_knowledge.models import (
    Activity,
    ActivityEvent,
    Attribute,
    Extraction,
    Person,
    Place,
    Travel,
)

__all__ = ["RuleBasedExtractor"]

# Pronoun -> gender. None means the pronoun carries no gender information,
# so we resolve the referent but assert nothing about gender.
PRONOUN_GENDER: dict[str, str | None] = {
    "she": "Female", "her": "Female", "hers": "Female",
    "he": "Male", "him": "Male", "his": "Male",
    "they": None, "them": None, "their": None,
}

TRAVEL_VERBS = {
    "went", "goes", "go", "travelled", "traveled", "travels", "drove",
    "drives", "walked", "walks", "rode", "flew", "flies", "headed", "heads",
    "returned", "returns", "arrived", "arrives",
}

PLACE_TYPES = {
    "bank": "Bank", "school": "School", "hospital": "Hospital",
    "store": "Store", "shop": "Shop", "office": "Office",
    "library": "Library", "restaurant": "Restaurant", "park": "Park",
    "airport": "Airport", "market": "Market", "church": "Church",
    "gym": "Gym", "hotel": "Hotel", "museum": "Museum",
}

# Irregular past tense -> base form. A real pipeline uses a lemmatiser.
LEMMAS = {
    "withdrew": "withdraw", "took": "take", "made": "make", "met": "meet",
    "bought": "buy", "sold": "sell", "paid": "pay", "sent": "send",
    "gave": "give", "spoke": "speak", "wrote": "write", "read": "read",
    "left": "leave", "found": "find", "saw": "see", "ate": "eat",
}

# Activity type -> attribute names that activity is expected to have. Values
# stay None unless the text supplies them, which is the point: the graph
# records that an amount exists and is unknown.
ACTIVITY_ATTRIBUTES = {
    "withdraw money": ["amount"],
    "deposit money": ["amount"],
    "transfer money": ["amount"],
    "pay money": ["amount"],
    "buy": ["amount"],
}

DETERMINERS = {"the", "a", "an", "some", "any", "his", "her", "their", "my", "our"}

# Normalise object nouns so that "withdrew some money" and "withdrew 500
# dollars" describe the same activity type rather than two unrelated ones.
NOUN_SYNONYMS = {
    "dollars": "money", "dollar": "money", "euros": "money", "euro": "money",
    "pounds": "money", "pound": "money", "cash": "money", "funds": "money",
}

# Capitalised function words that start sentences; never person names.
NON_NAMES = DETERMINERS | {
    "this", "that", "these", "those", "there", "it", "its", "no", "all",
    "both", "each", "every", "after", "before", "when", "while", "however",
    "if", "as", "at", "in", "on", "for", "we", "they", "you", "i",
}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")
_MONEY_RE = re.compile(r"(?:[$£€]\s?([\d,]+(?:\.\d+)?)|\b([\d,]+(?:\.\d+)?)\s+(dollars|euros|pounds))")


def _lemma(verb: str) -> str:
    """Crude stemmer: irregulars by table, then suffix stripping.

    Restoring a dropped "e" ("escalat" -> "escalate") is not attempted --
    "audited" -> "audit" and "escalated" -> "escalate" are indistinguishable
    without a lexicon, and guessing wrong is worse than under-stemming. This
    is a known limitation of the baseline; the LLM extractor does not have it.
    """
    low = verb.lower()
    if low in LEMMAS:
        return LEMMAS[low]
    for suffix, cut in (("ied", 3), ("ed", 2), ("es", 2), ("s", 1)):
        if low.endswith(suffix) and len(low) - cut >= 3:
            base = low[: len(low) - cut]
            if suffix == "ied":
                return base + "y"
            # "flagged" -> "flagg" -> "flag"; leave "ss"/"ll"/"ff" alone.
            if (
                len(base) >= 3
                and base[-1] == base[-2]
                and base[-1] not in "slfz"
                and base[-1] not in "aeiou"
            ):
                base = base[:-1]
            return base
    return low


class RuleBasedExtractor:
    """Extract entities and relations using lexicons and simple coreference."""

    def extract(self, text: str, doc_id: str) -> Extraction:
        result = Extraction(doc_id=doc_id, text=text)
        people: dict[str, Person] = {}
        last_person: Person | None = None
        last_place: Place | None = None

        for sentence in _SENTENCE_RE.split(text.strip()):
            tokens = _WORD_RE.findall(sentence)
            if not tokens:
                continue

            subject, rest = self._resolve_subject(tokens, people, last_person)
            if subject is None:
                continue
            last_person = subject

            if not rest:
                continue
            verb, obj_tokens = rest[0], rest[1:]

            if verb.lower() in TRAVEL_VERBS:
                place = self._parse_place(obj_tokens)
                if place is not None:
                    last_place = place
                    result.travels.append(Travel(person=subject, place=place))
            else:
                activity = self._parse_activity(verb, obj_tokens, sentence)
                if activity is not None:
                    result.activities.append(
                        ActivityEvent(person=subject, activity=activity, place=last_place)
                    )

        result.people = list(people.values())
        result.places = self._collect_places(result)
        return result

    # -- subject resolution -------------------------------------------------

    def _resolve_subject(
        self,
        tokens: list[str],
        people: dict[str, Person],
        last_person: Person | None,
    ) -> tuple[Person | None, list[str]]:
        """Return the sentence's subject and the tokens after it.

        A pronoun resolves to the most recently mentioned person and may
        contribute gender to that person -- this is what turns "She withdrew
        some money" into a fact about Mary.
        """
        head = tokens[0]
        low = head.lower()

        if low in PRONOUN_GENDER:
            if last_person is None:
                return None, []
            gender = PRONOUN_GENDER[low]
            if gender and not last_person.gender:
                last_person.gender = gender
            return last_person, tokens[1:]

        if head[:1].isupper() and low not in NON_NAMES:
            # Consume a run of capitalised tokens as one name ("Mary Jane").
            span = 1
            while span < len(tokens) and tokens[span][:1].isupper():
                span += 1
            name = " ".join(tokens[:span])
            person = people.get(name)
            if person is None:
                person = Person(name=name)
                people[name] = person
            return person, tokens[span:]

        return None, []

    # -- clause parsing -----------------------------------------------------

    def _parse_place(self, tokens: list[str]) -> Place | None:
        """Find a place type after a travel verb: "to the bank"."""
        for i, token in enumerate(tokens):
            if token.lower() == "to":
                for candidate in tokens[i + 1 :]:
                    low = candidate.lower()
                    if low in DETERMINERS:
                        continue
                    if low in PLACE_TYPES:
                        return Place(type=PLACE_TYPES[low])
                    if candidate[:1].isupper():
                        return Place(name=candidate)
                    break
        return None

    def _parse_activity(
        self, verb: str, obj_tokens: list[str], sentence: str
    ) -> Activity | None:
        obj = [t for t in obj_tokens if t.lower() not in DETERMINERS]
        if not obj:
            return None

        noun = obj[0].lower()
        activity_type = f"{_lemma(verb)} {NOUN_SYNONYMS.get(noun, noun)}"
        attributes = [
            Attribute(name=name)
            for name in ACTIVITY_ATTRIBUTES.get(activity_type, [])
        ]

        # If the text actually states an amount, fill the value in.
        money = _MONEY_RE.search(sentence)
        if money:
            amount = money.group(1) or money.group(2)
            unit = money.group(3)
            for attribute in attributes:
                if attribute.name == "amount":
                    attribute.value = amount.replace(",", "")
                    attribute.unit = unit

        return Activity(type=activity_type, attributes=attributes)

    @staticmethod
    def _collect_places(result: Extraction) -> list[Place]:
        seen: list[Place] = []
        for place in [t.place for t in result.travels] + [
            a.place for a in result.activities if a.place
        ]:
            if not any(p is place for p in seen):
                seen.append(place)
        return seen
