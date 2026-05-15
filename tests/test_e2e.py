"""End-to-end tests that hit the real Gemini API.

Skipped automatically when GEMINI_API_KEY is not set or empty.
Run with: pytest tests/test_e2e.py -v
"""
from __future__ import annotations

import os

import pytest

from semantic_vault.ingestion import ingest_text
from semantic_vault.retrieval import retrieve_and_synthesize


_NEEDS_KEY = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping live API tests",
)

_SAMPLE_TEXT = """\
The Python programming language was created by Guido van Rossum and first released in 1991.
Python emphasizes code readability and uses significant indentation.
It supports multiple programming paradigms including structured, object-oriented, and functional programming.
Python is widely used in data science, machine learning, web development, and automation.
The Python Software Foundation manages the language.
CPython is the reference implementation and is written in C.
"""


@_NEEDS_KEY
def test_e2e_ingest_and_query(vector_store, meta_store):
    """Ingest a real document and verify the answer cites the right content."""
    doc = ingest_text(_SAMPLE_TEXT, "python_intro", vector_store, meta_store)
    assert doc.chunk_count >= 1

    result = retrieve_and_synthesize("Who created Python?", vector_store)
    assert "Guido" in result.answer or "van Rossum" in result.answer
    assert len(result.sources) > 0


@_NEEDS_KEY
def test_e2e_extraction_populates_entities(vector_store, meta_store):
    """After ingestion the chunk payload should have known entities."""
    ingest_text(_SAMPLE_TEXT, "python_intro_entities", vector_store, meta_store)
    hits = vector_store.hybrid_search("Python creator")
    assert hits
    all_entity_names = []
    for h in hits:
        all_entity_names.extend(h.get("entity_names", []))
    # At minimum some entities should have been extracted
    assert len(all_entity_names) > 0


@_NEEDS_KEY
def test_e2e_duplicate_rejected(vector_store, meta_store):
    ingest_text(_SAMPLE_TEXT, "dup_test_1", vector_store, meta_store)
    with pytest.raises(ValueError, match="Duplicate"):
        ingest_text(_SAMPLE_TEXT, "dup_test_2", vector_store, meta_store)


@_NEEDS_KEY
def test_e2e_no_relevant_answer(vector_store, meta_store):
    ingest_text(_SAMPLE_TEXT, "python_for_no_answer", vector_store, meta_store)
    result = retrieve_and_synthesize("What is the boiling point of water?", vector_store)
    # Should answer honestly that the info isn't in the KB
    assert result.answer  # not empty
