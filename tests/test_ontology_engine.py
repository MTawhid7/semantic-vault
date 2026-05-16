"""Unit tests for OntologyEngine and SchemaRegistry."""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from semantic_vault.ontology_engine import (
    MIN_CLUSTER_SIZE,
    OntologyEngine,
    SchemaRegistry,
    _union_find_cluster,
)


# ---------------------------------------------------------------------------
# SchemaRegistry
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry(tmp_path) -> SchemaRegistry:
    return SchemaRegistry(db_path=str(tmp_path / "reg.db"))


def test_registry_seeded_with_base_types(registry):
    types = {t["type_name"] for t in registry.list_types()}
    for base in ["Person", "Organization", "Location", "Event", "Concept"]:
        assert base in types


def test_registry_register_new_type(registry):
    result = registry.register_type("ResearchInstitution", "Organization", ["MIT", "Stanford"])
    assert result is True
    types = {t["type_name"] for t in registry.list_types()}
    assert "ResearchInstitution" in types


def test_registry_duplicate_type_returns_false(registry):
    registry.register_type("NewType", "Concept")
    assert registry.register_type("NewType", "Concept") is False


def test_registry_version_increments(registry):
    v0 = registry.current_version()
    registry.register_type("TypeA", "Concept")
    v1 = registry.current_version()
    assert v1 > v0


def test_registry_preserves_example_entities(registry):
    import json
    registry.register_type("Drug", "Concept", ["Aspirin", "Ibuprofen"])
    row = next(t for t in registry.list_types() if t["type_name"] == "Drug")
    examples = json.loads(row["example_entities"])
    assert "Aspirin" in examples


# ---------------------------------------------------------------------------
# Union-find clustering helper
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, seed: int):
    """Deterministic unit vector."""
    import random as _r
    rng = _r.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) + 1e-9
    return [x / norm for x in v]


def test_cluster_identical_vectors_together():
    import numpy as np
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.stack([vec, vec, vec])  # all identical → one cluster
    clusters = _union_find_cluster(embeddings, threshold=0.99)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_orthogonal_vectors_separately():
    import numpy as np
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    clusters = _union_find_cluster(np.stack([e1, e2, e3]), threshold=0.5)
    assert len(clusters) == 3  # all separate


def test_cluster_threshold_boundary():
    import numpy as np
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.9, 0.1], dtype=np.float32)
    b /= np.linalg.norm(b)
    sim = float(np.dot(a, b))
    clusters_above = _union_find_cluster(np.stack([a, b]), threshold=sim - 0.01)
    clusters_below = _union_find_cluster(np.stack([a, b]), threshold=sim + 0.01)
    assert len(clusters_above) == 1   # merged
    assert len(clusters_below) == 2   # separate


# ---------------------------------------------------------------------------
# OntologyEngine
# ---------------------------------------------------------------------------

def _mock_entity_index(concept_names: list[str]) -> MagicMock:
    """Returns an EntityIndex mock whose scroll returns Concept entities."""
    import numpy as np

    # Make all concepts point in the same direction → cluster together
    base_vec = np.ones(32, dtype=np.float32)
    base_vec /= np.linalg.norm(base_vec)

    points = []
    for i, name in enumerate(concept_names):
        p = MagicMock()
        p.id = f"id-{i}"
        p.payload = {"canonical_id": f"id-{i}", "name": name, "type": "Concept"}
        # Slight noise so they're not identical but still highly similar
        noise = np.random.default_rng(i).random(32).astype(np.float32) * 0.01
        vec = base_vec + noise
        vec /= np.linalg.norm(vec)
        p.vector = {"dense": vec.tolist()}
        points.append(p)

    qdrant = MagicMock()
    qdrant.scroll.return_value = (points, None)  # None = no next page
    idx = MagicMock()
    idx._qdrant = qdrant
    return idx


def _mock_gemini_type(name: str) -> MagicMock:
    response = MagicMock()
    response.text = name
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def test_ontology_engine_proposes_type_for_large_cluster(registry, tmp_path):
    names = [f"chemical_{i}" for i in range(MIN_CLUSTER_SIZE + 2)]
    idx = _mock_entity_index(names)
    engine = OntologyEngine(
        entity_index=idx,
        schema_registry=registry,
        gemini_client=_mock_gemini_type("ChemicalCompound"),
        min_cluster_size=MIN_CLUSTER_SIZE,
        cluster_threshold=0.5,  # low threshold so all similar vectors cluster
    )
    new_types = engine.run_evolution()
    assert "ChemicalCompound" in new_types
    assert any(t["type_name"] == "ChemicalCompound" for t in registry.list_types())


def test_ontology_engine_skips_small_clusters(registry):
    names = ["entity_a", "entity_b"]  # only 2 — below MIN_CLUSTER_SIZE
    idx = _mock_entity_index(names)
    engine = OntologyEngine(
        entity_index=idx,
        schema_registry=registry,
        gemini_client=_mock_gemini_type("SomeType"),
        min_cluster_size=5,
    )
    new_types = engine.run_evolution()
    assert new_types == []


def test_ontology_engine_does_not_duplicate_types(registry, tmp_path):
    names = [f"node_{i}" for i in range(MIN_CLUSTER_SIZE + 1)]
    idx = _mock_entity_index(names)
    client = _mock_gemini_type("UniqueType")
    engine = OntologyEngine(idx, registry, client, cluster_threshold=0.5)
    engine.run_evolution()
    new_types = engine.run_evolution()  # second run — type already registered
    assert "UniqueType" not in new_types
