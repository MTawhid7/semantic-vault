"""Shared fixtures for Semantic Vault tests (Phase 1 + Phase 2)."""
from __future__ import annotations

import math
import uuid
from unittest.mock import MagicMock

import pytest

from semantic_vault.entity_resolver import EntityResolver
from semantic_vault.storage import EntityIndex, MetadataStore, VectorStore

_DIM = 3072  # must match settings.gemini_embed_dim


# ---------------------------------------------------------------------------
# Mock embedding function
# ---------------------------------------------------------------------------

def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic unit vectors — same text always returns same vector."""
    import random
    results = []
    for text in texts:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        vec = [rng.gauss(0, 1) for _ in range(_DIM)]
        norm = math.sqrt(sum(x * x for x in vec)) + 1e-9
        results.append([x / norm for x in vec])
    return results


# ---------------------------------------------------------------------------
# Phase 1 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vector_store() -> VectorStore:
    return VectorStore(url=":memory:", embed_fn=_mock_embed)


@pytest.fixture()
def meta_store(tmp_path) -> MetadataStore:
    return MetadataStore(db_path=str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# Phase 2 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def entity_index(vector_store) -> EntityIndex:
    """Real EntityIndex backed by the in-memory vector_store."""
    return EntityIndex(vector_store)


@pytest.fixture()
def mock_graph_store() -> MagicMock:
    """Fully mocked GraphStore — no Neo4j required."""
    gs = MagicMock()
    gs.upsert_entity = MagicMock()
    gs.add_alias = MagicMock()
    gs.upsert_relationship = MagicMock()
    gs.traverse = MagicMock(return_value=[])
    gs.find_entity = MagicMock(return_value=None)
    gs.text_to_cypher_query = MagicMock(return_value=([], ""))
    gs.entity_count = MagicMock(return_value=0)
    gs.relationship_count = MagicMock(return_value=0)
    return gs


@pytest.fixture()
def entity_resolver(entity_index) -> EntityResolver:
    """EntityResolver with real EntityIndex but mocked LLM (always says 'different')."""
    return EntityResolver(
        entity_index,
        _llm_fn=lambda name_a, name_b, ctx: False,  # conservative: never merge via LLM
    )


def neo4j_available() -> bool:
    """Return True if a local Neo4j instance is reachable."""
    try:
        from neo4j import GraphDatabase  # type: ignore[import-untyped]
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password")
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


needs_neo4j = pytest.mark.skipif(not neo4j_available(), reason="Neo4j not available")
