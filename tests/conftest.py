"""Shared fixtures for Semantic Vault tests."""
from __future__ import annotations

import math
import pytest

from semantic_vault.storage import MetadataStore, VectorStore

_DIM = 3072  # must match settings.gemini_embed_dim


# ---------------------------------------------------------------------------
# Mock embedding function — avoids any real API call during tests
# ---------------------------------------------------------------------------

def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Return deterministic unit vectors of dimension _DIM.

    Identical texts get identical vectors (hash-seeded), so semantic
    similarity tests behave predictably without touching the network.
    """
    import random
    results = []
    for text in texts:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        vec = [rng.gauss(0, 1) for _ in range(_DIM)]
        norm = math.sqrt(sum(x * x for x in vec)) + 1e-9
        results.append([x / norm for x in vec])
    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vector_store() -> VectorStore:
    """In-memory Qdrant with a mock embedding function (no network calls)."""
    return VectorStore(url=":memory:", embed_fn=_mock_embed)


@pytest.fixture()
def meta_store(tmp_path) -> MetadataStore:
    """SQLite store in a temporary directory."""
    return MetadataStore(db_path=str(tmp_path / "test.db"))
