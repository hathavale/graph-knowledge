"""The extractor seam.

Everything downstream depends on `Extraction`, never on how it was produced,
so the baseline extractor can be swapped for an LLM-backed one without the
store or the tests changing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from graph_knowledge.models import Extraction

__all__ = ["Extractor"]


@runtime_checkable
class Extractor(Protocol):
    def extract(self, text: str, doc_id: str) -> Extraction:
        """Turn a passage of text into entities and relationships."""
        ...
