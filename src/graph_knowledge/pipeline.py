"""Wire an extractor to a store."""

from __future__ import annotations

from graph_knowledge.extraction.base import Extractor
from graph_knowledge.models import Extraction
from graph_knowledge.store.base import GraphStore

__all__ = ["Pipeline"]


class Pipeline:
    def __init__(self, extractor: Extractor, store: GraphStore) -> None:
        self._extractor = extractor
        self._store = store

    def ingest(self, text: str, doc_id: str) -> Extraction:
        """Extract and persist one document, returning what was extracted.

        Safe to re-run: writes MERGE on deterministic keys, so ingesting the
        same document twice leaves the graph unchanged.
        """
        extraction = self._extractor.extract(text, doc_id)
        self._store.write(extraction)
        return extraction
