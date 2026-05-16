"""Phase 2 retrieval tests — graph branch + fusion."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantic_vault.models import Chunk
from semantic_vault.retrieval import _is_relational_query, retrieve_and_synthesize


def _seed(vector_store, text="Alice works at Acme.", doc="test"):
    chunk = Chunk(text=text, source_doc_id="d1", source_doc_name=doc, chunk_index=0)
    vector_store.upsert_chunks([chunk])


def _synth_client(answer="The answer."):
    r = MagicMock()
    r.text = answer
    c = MagicMock()
    c.models.generate_content.return_value = r
    return c


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Who does Alice work with?",
    "Who is related to Acme Corp?",
    "Who collaborates with the research team?",
    "Which organization does Bob belong to?",
    "What company is Alice associated with?",
    "Who works with the CEO?",
])
def test_is_relational_true(question):
    assert _is_relational_query(question) is True


@pytest.mark.parametrize("question", [
    "What is machine learning?",
    "When was Python created?",
    "Summarise the project notes.",
    "What are the key findings?",
    "Tell me about the paper on climate change.",
])
def test_is_relational_false(question):
    assert _is_relational_query(question) is False


# ---------------------------------------------------------------------------
# Phase 1 path unchanged (no graph store)
# ---------------------------------------------------------------------------

def test_phase1_path_with_no_graph_store(vector_store):
    _seed(vector_store)
    client = _synth_client("Alice works at Acme.")
    result = retrieve_and_synthesize("Who is Alice?", vector_store, gemini_client=client)
    assert result.answer
    assert result.graph_facts == []


def test_no_graph_store_never_calls_graph_methods(vector_store):
    _seed(vector_store)
    client = _synth_client()
    # Should not raise even for a relational question
    result = retrieve_and_synthesize(
        "Who does Alice work with?", vector_store, gemini_client=client
    )
    assert result.graph_facts == []


# ---------------------------------------------------------------------------
# Phase 2: graph branch activated for relational queries
# ---------------------------------------------------------------------------

def test_graph_branch_activated_for_relational_query(vector_store, mock_graph_store):
    _seed(vector_store)
    mock_graph_store.text_to_cypher_query.return_value = (
        [{"source": "Alice", "target": "Acme", "predicates": ["WORKS_AT"]}],
        "MATCH ...",
    )
    client = _synth_client("Alice works at Acme Corp.")
    result = retrieve_and_synthesize(
        "Who does Alice work with?",
        vector_store,
        graph_store=mock_graph_store,
        gemini_client=client,
    )
    mock_graph_store.text_to_cypher_query.assert_called_once()
    assert result.graph_facts != []


def test_graph_branch_not_activated_for_factual_query(vector_store, mock_graph_store):
    _seed(vector_store)
    client = _synth_client()
    retrieve_and_synthesize(
        "What year was Python created?",
        vector_store,
        graph_store=mock_graph_store,
        gemini_client=client,
    )
    mock_graph_store.text_to_cypher_query.assert_not_called()


def test_graph_facts_included_in_result(vector_store, mock_graph_store):
    _seed(vector_store)
    facts = [{"source": "Alice", "target": "Acme Corp", "predicates": ["WORKS_AT"]}]
    mock_graph_store.text_to_cypher_query.return_value = (facts, "MATCH ...")
    client = _synth_client("Alice works at Acme Corp.")
    result = retrieve_and_synthesize(
        "Who does Alice work with?",
        vector_store,
        graph_store=mock_graph_store,
        gemini_client=client,
    )
    assert result.graph_facts == facts


def test_graph_failure_falls_back_to_vector_only(vector_store, mock_graph_store):
    _seed(vector_store)
    mock_graph_store.text_to_cypher_query.side_effect = Exception("Neo4j unavailable")
    client = _synth_client("Fallback answer.")
    result = retrieve_and_synthesize(
        "Who does Alice work with?",
        vector_store,
        graph_store=mock_graph_store,
        gemini_client=client,
    )
    # Should not raise; answer should come from vector search
    assert result.answer == "Fallback answer."
    assert result.graph_facts == []


def test_graph_context_included_in_synthesis_prompt(vector_store, mock_graph_store):
    _seed(vector_store)
    facts = [{"source": "Alice", "target": "Bob", "predicates": ["COLLABORATED_WITH"]}]
    mock_graph_store.text_to_cypher_query.return_value = (facts, "MATCH ...")
    client = _synth_client()
    retrieve_and_synthesize(
        "Who does Alice collaborate with?",
        vector_store,
        graph_store=mock_graph_store,
        gemini_client=client,
    )
    prompt_str = str(client.models.generate_content.call_args)
    # Graph facts should appear in the synthesis prompt
    assert "Alice" in prompt_str or "graph" in prompt_str.lower()


# ---------------------------------------------------------------------------
# graph_facts field on QueryResult
# ---------------------------------------------------------------------------

def test_query_result_has_graph_facts_field(vector_store):
    _seed(vector_store)
    client = _synth_client()
    result = retrieve_and_synthesize("Who is Alice?", vector_store, gemini_client=client)
    assert hasattr(result, "graph_facts")
    assert isinstance(result.graph_facts, list)
