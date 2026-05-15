"""Tests for the retrieval + synthesis pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantic_vault.models import Chunk, ExtractionOutput, ExtractedEntity
from semantic_vault.retrieval import retrieve_and_synthesize


def _seed_store(vector_store, docs: list[tuple[str, str, str]]) -> None:
    """docs = [(text, doc_id, doc_name)]"""
    chunks = [
        Chunk(text=text, source_doc_id=doc_id, source_doc_name=doc_name, chunk_index=i)
        for i, (text, doc_id, doc_name) in enumerate(docs)
    ]
    vector_store.upsert_chunks(chunks)


def _mock_synthesis_client(answer: str = "The answer is 42.") -> MagicMock:
    response = MagicMock()
    response.text = answer
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------------


def test_retrieve_returns_query_result(vector_store):
    _seed_store(vector_store, [("Paris is in France.", "d1", "geo")])
    client = _mock_synthesis_client("Paris is the capital of France.")
    result = retrieve_and_synthesize("Where is Paris?", vector_store, gemini_client=client)
    assert result.answer == "Paris is the capital of France."


def test_retrieve_sources_populated(vector_store):
    _seed_store(vector_store, [("Sample text.", "d1", "sample_doc")])
    client = _mock_synthesis_client()
    result = retrieve_and_synthesize("sample", vector_store, gemini_client=client)
    assert len(result.sources) > 0
    assert result.sources[0]["doc_name"] == "sample_doc"


def test_retrieve_empty_store_returns_no_info_message(vector_store):
    client = _mock_synthesis_client()
    result = retrieve_and_synthesize("anything", vector_store, gemini_client=client)
    assert "No relevant information" in result.answer
    assert result.sources == []
    # Should not have called Gemini at all
    client.models.generate_content.assert_not_called()


def test_retrieve_top_k_limits_sources(vector_store):
    docs = [(f"Doc {i} content.", f"d{i}", f"doc{i}") for i in range(10)]
    _seed_store(vector_store, docs)
    client = _mock_synthesis_client()
    result = retrieve_and_synthesize("content", vector_store, top_k=3, gemini_client=client)
    assert len(result.sources) <= 3


def test_retrieve_entities_mentioned_populated(vector_store):
    chunk = Chunk(
        text="Alice and Bob collaborated on the project.",
        source_doc_id="d1",
        source_doc_name="collab",
        chunk_index=0,
        extraction=ExtractionOutput(
            entities=[
                ExtractedEntity(name="Alice", type="Person", confidence=0.9),
                ExtractedEntity(name="Bob", type="Person", confidence=0.9),
            ],
            key_facts=[],
            summary="",
            topics=[],
        ),
    )
    vector_store.upsert_chunks([chunk])
    client = _mock_synthesis_client("They collaborated.")
    result = retrieve_and_synthesize("who collaborated", vector_store, gemini_client=client)
    assert "Alice" in result.entities_mentioned or "Bob" in result.entities_mentioned


def test_retrieve_synthesis_client_called_with_chunks_and_question(vector_store):
    _seed_store(vector_store, [("Einstein discovered relativity.", "d1", "physics")])
    client = _mock_synthesis_client()
    retrieve_and_synthesize("What did Einstein discover?", vector_store, gemini_client=client)
    call_args = client.models.generate_content.call_args
    prompt_contents = call_args.kwargs.get("contents") or call_args.args[0] if call_args.args else ""
    assert "Einstein" in str(prompt_contents) or "Einstein" in str(call_args)


def test_retrieve_sources_have_required_keys(vector_store):
    _seed_store(vector_store, [("Test content.", "d1", "test")])
    client = _mock_synthesis_client()
    result = retrieve_and_synthesize("test", vector_store, gemini_client=client)
    for source in result.sources:
        for key in ("rank", "chunk_id", "doc_name", "chunk_index", "score", "snippet"):
            assert key in source
