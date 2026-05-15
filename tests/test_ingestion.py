"""Tests for the ingestion pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantic_vault.ingestion import ingest_text
from semantic_vault.models import ExtractionOutput, ExtractedEntity


def _mock_gemini(summary: str = "A test document.") -> MagicMock:
    extraction = ExtractionOutput(
        entities=[ExtractedEntity(name="Test", type="Concept", confidence=0.8)],
        key_facts=["This is a fact."],
        summary=summary,
        topics=["test"],
    )
    response = MagicMock()
    response.parsed = extraction
    response.text = extraction.model_dump_json()
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------------


def test_ingest_returns_source_document(vector_store, meta_store):
    doc = ingest_text(
        "Alice studies machine learning at MIT.",
        "test_doc",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(),
    )
    assert doc.name == "test_doc"
    assert doc.chunk_count >= 1


def test_ingest_stores_chunks_in_vector_store(vector_store, meta_store):
    ingest_text(
        "Paris is the capital of France. It has the Eiffel Tower.",
        "geo_doc",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(),
    )
    assert vector_store.count() >= 1


def test_ingest_records_document_in_meta_store(vector_store, meta_store):
    ingest_text(
        "Some content here.",
        "meta_test",
        vector_store,
        meta_store,
        gemini_client=_mock_gemini(),
    )
    assert meta_store.count() == 1
    docs = meta_store.list_documents()
    assert docs[0]["name"] == "meta_test"


def test_ingest_duplicate_raises_value_error(vector_store, meta_store):
    content = "Identical content that will be ingested twice."
    ingest_text(content, "first", vector_store, meta_store, gemini_client=_mock_gemini())
    with pytest.raises(ValueError, match="Duplicate"):
        ingest_text(content, "second", vector_store, meta_store, gemini_client=_mock_gemini())


def test_ingest_skip_extraction_still_stores(vector_store, meta_store):
    doc = ingest_text(
        "Content without LLM extraction.",
        "no_extract",
        vector_store,
        meta_store,
        skip_extraction=True,
    )
    assert vector_store.count() >= 1
    assert meta_store.count() == 1
    assert doc.chunk_count >= 1


def test_ingest_extraction_failure_is_non_fatal(vector_store, meta_store):
    """If extraction throws, ingestion still completes (best-effort)."""
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("API down")

    doc = ingest_text(
        "Content here." * 10,
        "error_doc",
        vector_store,
        meta_store,
        gemini_client=client,
    )
    # Document should still be ingested, just without extraction metadata
    assert doc.chunk_count >= 1
    assert meta_store.count() == 1


def test_ingest_long_document_produces_multiple_chunks(vector_store, meta_store):
    # ~3 000 chars → should produce multiple 1600-char chunks
    long_text = " ".join([f"This is sentence {i} of the long document." for i in range(80)])
    doc = ingest_text(long_text, "long_doc", vector_store, meta_store, gemini_client=_mock_gemini())
    assert doc.chunk_count > 1
    assert vector_store.count() == doc.chunk_count


def test_two_different_documents_both_stored(vector_store, meta_store):
    ingest_text("First doc content.", "doc_a", vector_store, meta_store, gemini_client=_mock_gemini())
    ingest_text("Second doc content.", "doc_b", vector_store, meta_store, gemini_client=_mock_gemini())
    assert meta_store.count() == 2
