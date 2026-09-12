"""Wire an extractor to a store, with the vocabulary in the loop."""

from __future__ import annotations

from graph_knowledge.extraction.base import Extractor
from graph_knowledge.models import Ticket, TicketExtraction
from graph_knowledge.store.base import GraphStore
from graph_knowledge.vocabulary import TermKind, Vocabulary

__all__ = ["Pipeline"]


class Pipeline:
    def __init__(
        self,
        extractor: Extractor,
        store: GraphStore,
        vocabulary: Vocabulary | None = None,
    ) -> None:
        self._extractor = extractor
        self._store = store
        self._vocabulary = vocabulary or Vocabulary()

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def ingest(self, text: str, ticket: Ticket | str) -> TicketExtraction:
        """Extract, record vocabulary observations, persist.

        Observing terms *before* the write means a topic promoted by this
        ticket is canonical from the next ticket onward -- the vocabulary
        grows as evidence accumulates rather than by anyone curating up front.
        """
        if isinstance(ticket, str):
            ticket = Ticket(id=ticket)

        extraction = self._extractor.extract(text, ticket)

        for topic in extraction.topics:
            self._vocabulary.observe(TermKind.TOPIC, topic.name, ticket.id)
        for fact in extraction.org_facts:
            if fact.relation.value == "HAS_FUNCTION":
                self._vocabulary.observe(TermKind.FUNCTION, fact.target, ticket.id)
            elif fact.relation.value == "MEMBER_OF":
                self._vocabulary.observe(TermKind.TEAM, fact.target, ticket.id)

        self._store.write(extraction)
        self._store.save_vocabulary(self._vocabulary.all_terms())
        return extraction
