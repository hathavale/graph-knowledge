"""The extractor seam."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from graph_knowledge.models import Ticket, TicketExtraction

__all__ = ["Extractor"]


@runtime_checkable
class Extractor(Protocol):
    def extract(self, text: str, ticket: Ticket | str) -> TicketExtraction:
        """Turn ticket text into people, topics, participation and org facts."""
        ...
