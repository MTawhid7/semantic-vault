"""Tests for VectorStore and MetadataStore."""
from __future__ import annotations

import pytest

from semantic_vault.models import (
    Chunk,
    ExtractionOutput,
    ExtractedEntity,
    SourceDocument,
)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


def test_vector_store_starts_empty(vector_store):
    assert vector_store.count() == 0


def test_upsert_single_chunk(vector_store):
    chunk = Chunk(text="The Eiffel Tower is in Paris.", source_doc_id="d1", source_doc_name="doc1", chunk_index=0)
    vector_store.upsert_chunks([chunk])
    assert vector_store.count() == 1


def test_upsert_multiple_chunks(vector_store):
    chunks = [
        Chunk(text=f"Chunk {i}.", source_doc_id="d1", source_doc_name="doc1", chunk_index=i)
        for i in range(5)
    ]
    vector_store.upsert_chunks(chunks)
    assert vector_store.count() == 5


def test_upsert_empty_list_is_noop(vector_store):
    vector_store.upsert_chunks([])
    assert vector_store.count() == 0


def test_hybrid_search_returns_results(vector_store):
    chunks = [
        Chunk(text="Paris is the capital of France.", source_doc_id="d1", source_doc_name="France", chunk_index=0),
        Chunk(text="Berlin is the capital of Germany.", source_doc_id="d2", source_doc_name="Germany", chunk_index=0),
    ]
    vector_store.upsert_chunks(chunks)
    results = vector_store.hybrid_search("capital city")
    assert len(results) > 0


def test_hybrid_search_result_has_expected_keys(vector_store):
    chunk = Chunk(text="Sample text.", source_doc_id="d1", source_doc_name="doc1", chunk_index=0)
    vector_store.upsert_chunks([chunk])
    results = vector_store.hybrid_search("sample")
    assert results
    hit = results[0]
    for key in ("id", "score", "text", "source_doc_name", "chunk_index"):
        assert key in hit


def test_hybrid_search_top_k_respected(vector_store):
    chunks = [
        Chunk(text=f"Document about topic {i}.", source_doc_id=f"d{i}", source_doc_name=f"doc{i}", chunk_index=0)
        for i in range(10)
    ]
    vector_store.upsert_chunks(chunks)
    results = vector_store.hybrid_search("topic", top_k=3)
    assert len(results) <= 3


def test_hybrid_search_empty_store_returns_empty(vector_store):
    results = vector_store.hybrid_search("anything")
    assert results == []


def test_upsert_chunk_with_extraction(vector_store):
    extraction = ExtractionOutput(
        entities=[ExtractedEntity(name="Paris", type="Location", confidence=0.98)],
        key_facts=["Paris is in France."],
        summary="About Paris.",
        topics=["paris", "france"],
    )
    chunk = Chunk(
        text="Paris is in France.",
        source_doc_id="d1",
        source_doc_name="geo",
        chunk_index=0,
        extraction=extraction,
    )
    vector_store.upsert_chunks([chunk])
    results = vector_store.hybrid_search("Paris")
    assert results
    assert "Paris" in results[0].get("entity_names", [])


def test_entity_type_filter(vector_store):
    chunks = [
        Chunk(
            text="Alice is a scientist.",
            source_doc_id="d1", source_doc_name="people", chunk_index=0,
            extraction=ExtractionOutput(
                entities=[ExtractedEntity(name="Alice", type="Person", confidence=0.9)],
                key_facts=[], summary="", topics=[],
            ),
        ),
        Chunk(
            text="OpenAI is an AI company.",
            source_doc_id="d2", source_doc_name="companies", chunk_index=0,
            extraction=ExtractionOutput(
                entities=[ExtractedEntity(name="OpenAI", type="Organization", confidence=0.95)],
                key_facts=[], summary="", topics=[],
            ),
        ),
    ]
    vector_store.upsert_chunks(chunks)
    results = vector_store.hybrid_search("person or company", entity_type_filter="Person")
    for r in results:
        assert "Person" in r.get("entity_types", [])


# ---------------------------------------------------------------------------
# MetadataStore
# ---------------------------------------------------------------------------


def test_meta_store_starts_empty(meta_store):
    assert meta_store.count() == 0


def test_save_and_retrieve_document(meta_store):
    doc = SourceDocument(name="test.txt", content_hash="abc123")
    meta_store.save_document(doc)
    assert meta_store.count() == 1
    docs = meta_store.list_documents()
    assert len(docs) == 1
    assert docs[0]["name"] == "test.txt"


def test_document_exists_by_hash(meta_store):
    doc = SourceDocument(name="test.txt", content_hash="hash_xyz")
    meta_store.save_document(doc)
    assert meta_store.document_exists("hash_xyz") is True
    assert meta_store.document_exists("other_hash") is False


def test_list_documents_ordered_by_date(meta_store):
    for i in range(3):
        meta_store.save_document(SourceDocument(name=f"doc{i}.txt", content_hash=f"hash{i}"))
    docs = meta_store.list_documents()
    assert len(docs) == 3


def test_save_duplicate_hash_is_idempotent(meta_store):
    doc = SourceDocument(name="dup.txt", content_hash="same_hash")
    meta_store.save_document(doc)
    meta_store.save_document(doc)  # second save; same hash = REPLACE
    assert meta_store.count() == 1
