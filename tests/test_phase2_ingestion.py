"""Phase 2 ingestion tests — entity resolution + graph writes.

All external dependencies (Gemini, Neo4j) are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantic_vault.ingestion import ingest_text
from semantic_vault.models import ExtractionOutput, ExtractedEntity, ExtractedRelationship


def _mock_gemini(extraction: ExtractionOutput | None = None) -> MagicMock:
    ex = extraction or ExtractionOutput(
        entities=[
            ExtractedEntity(name="Alice", type="Person", confidence=0.9),
            ExtractedEntity(name="Acme Corp", type="Organization", confidence=0.88),
        ],
        relationships=[
            ExtractedRelationship(
                subject="Alice", predicate="WORKS_AT", object="Acme Corp", confidence=0.85
            )
        ],
        key_facts=["Alice works at Acme Corp."],
        summary="About Alice at Acme.",
        topics=["employment"],
    )
    response = MagicMock()
    response.parsed = ex
    response.text = ex.model_dump_json()
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------------
# Phase 1 path still works (no graph args)
# ---------------------------------------------------------------------------

def test_phase1_path_unchanged(vector_store, meta_store):
    doc = ingest_text(
        "Alice works at Acme Corp.",
        "test",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(),
    )
    assert doc.chunk_count >= 1
    assert meta_store.count() == 1


# ---------------------------------------------------------------------------
# Phase 2: entity resolution is called for each entity
# ---------------------------------------------------------------------------

def test_phase2_resolver_called_for_each_entity(
    vector_store, meta_store, entity_index, mock_graph_store
):
    from semantic_vault.entity_resolver import EntityResolver
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)

    doc = ingest_text(
        "Alice works at Acme Corp.",
        "doc_er",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(),
        entity_index=entity_index,
        graph_store=mock_graph_store,
        entity_resolver=resolver,
    )
    # Both entities extracted → resolver should have called upsert_entity on graph_store
    assert mock_graph_store.upsert_entity.call_count >= 1


def test_phase2_relationship_written_to_graph(
    vector_store, meta_store, entity_index, mock_graph_store
):
    from semantic_vault.entity_resolver import EntityResolver
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)

    ingest_text(
        "Alice works at Acme Corp.",
        "rel_doc",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(),
        entity_index=entity_index,
        graph_store=mock_graph_store,
        entity_resolver=resolver,
    )
    # WORKS_AT relationship should have been written
    assert mock_graph_store.upsert_relationship.call_count >= 1
    call_kwargs = mock_graph_store.upsert_relationship.call_args
    # predicate should be normalised
    pred = call_kwargs[0][1] if call_kwargs[0] else call_kwargs[1].get("predicate", "")
    assert pred == "WORKS_AT" or "WORKS_AT" in str(call_kwargs)


def test_phase2_duplicate_entity_gets_alias(
    vector_store, meta_store, entity_index, mock_graph_store
):
    """Ingesting two documents about the same entity should call add_alias on the second."""
    from semantic_vault.entity_resolver import EntityResolver
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)

    ingest_text(
        "Alice Johnson works at Acme Corp.",
        "doc1",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(),
        entity_index=entity_index,
        graph_store=mock_graph_store,
        entity_resolver=resolver,
    )
    # Add Alice again with the exact same name → entity_index finds it → is_new=False
    from semantic_vault.models import ExtractionOutput, ExtractedEntity
    ex2 = ExtractionOutput(
        entities=[ExtractedEntity(name="Alice", type="Person", confidence=0.95)],
        relationships=[],
        key_facts=[],
        summary="",
        topics=[],
    )
    ingest_text(
        "Alice Johnson is also a researcher.",
        "doc2",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(ex2),
        entity_index=entity_index,
        graph_store=mock_graph_store,
        entity_resolver=resolver,
    )
    # Second encounter → should call add_alias (not upsert_entity again)
    assert mock_graph_store.add_alias.call_count >= 1


def test_phase2_graph_failure_is_non_fatal(
    vector_store, meta_store, entity_index
):
    """If the graph store raises, ingestion still completes."""
    bad_graph = MagicMock()
    bad_graph.upsert_entity.side_effect = RuntimeError("Neo4j down")
    bad_graph.add_alias.side_effect = RuntimeError("Neo4j down")

    from semantic_vault.entity_resolver import EntityResolver
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)

    doc = ingest_text(
        "Alice works somewhere.",
        "fault_doc",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(),
        entity_index=entity_index,
        graph_store=bad_graph,
        entity_resolver=resolver,
    )
    assert doc.chunk_count >= 1  # phase 1 still completed
    assert meta_store.count() == 1


def test_phase2_no_relationship_if_entity_missing(
    vector_store, meta_store, entity_index, mock_graph_store
):
    """Relationships whose endpoint wasn't in entities[] must not be written."""
    ex = ExtractionOutput(
        entities=[ExtractedEntity(name="Alice", type="Person", confidence=0.9)],
        relationships=[
            ExtractedRelationship(
                subject="Alice", predicate="WORKS_AT", object="Unknown Corp", confidence=0.7
            )
        ],
        key_facts=[],
        summary="",
        topics=[],
    )
    # Note: after _normalise in extractor, "Unknown Corp" is not in entities →
    # the relationship will be filtered out before even reaching ingestion.
    # But let's test the ingestion guard too.
    from semantic_vault.entity_resolver import EntityResolver
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)

    ingest_text(
        "Alice works somewhere.",
        "missing_ep",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(ex),
        entity_index=entity_index,
        graph_store=mock_graph_store,
        entity_resolver=resolver,
    )
    # "Unknown Corp" was not resolved → relationship must not be written
    mock_graph_store.upsert_relationship.assert_not_called()
