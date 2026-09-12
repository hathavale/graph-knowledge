"""Backend fixtures.

Parametrised over every backend available: the embedded one always runs, and
Neo4j joins when a server answers, so `pytest` is green with or without
`make up`.
"""

from __future__ import annotations

import pytest

from graph_knowledge.store.embedded_store import EmbeddedStore, embedded_available
from graph_knowledge.store.neo4j_store import Neo4jStore, neo4j_available


def _backends() -> list[str]:
    available = []
    if embedded_available():
        available.append("embedded")
    if neo4j_available():
        available.append("neo4j")
    return available or ["embedded"]


@pytest.fixture(params=_backends())
def store(request, tmp_path):
    backend = request.param
    if backend == "embedded":
        if not embedded_available():
            pytest.skip("ladybug not installed (pip install -e '.[embedded]')")
        instance = EmbeddedStore(str(tmp_path / "graph"))
    else:
        if not neo4j_available():
            pytest.skip("no Neo4j on bolt://localhost:7687 (run `make up`)")
        instance = Neo4jStore()

    instance.initialize()
    instance.reset()
    try:
        yield instance
    finally:
        instance.close()
