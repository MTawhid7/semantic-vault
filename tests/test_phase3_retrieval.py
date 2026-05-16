"""Phase 3 retrieval tests — query planner, parallel retrieval, RRF, reranking."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from semantic_vault.models import Chunk
from semantic_vault.query_planner import QueryPlan
from semantic_vault.retrieval import _is_relational_query, retrieve_and_synthesize
from semantic_vault.reranker import rrf_fuse


def _seed(vector_store, texts_and_names: list[tuple[str, str, str]]):
    chunks = [
        Chunk(text=t, source_doc_id=did, source_doc_name=dn, chunk_index=i)
        for i, (t, did, dn) in enumerate(texts_and_names)
    ]
    vector_store.upsert_chunks(chunks)


def _synth_client(answer: str = "Test answer.") -> MagicMock:
    r = MagicMock()
    r.text = answer
    c = MagicMock()
    c.models.generate_content.return_value = r
    return c


def _plan_client(plan: QueryPlan) -> MagicMock:
    from semantic_vault.query_planner import _GeminiQueryPlan
    schema_obj = _GeminiQueryPlan(
        intent=plan.intent,
        sub_questions=plan.sub_questions,
        strategies=plan.strategies,
        hop_depth=plan.hop_depth,
        needs_reranking=plan.needs_reranking,
    )
    r = MagicMock()
    r.parsed = schema_obj
    r.text = schema_obj.model_dump_json()
    c = MagicMock()
    c.models.generate_content.return_value = r
    return c


# ---------------------------------------------------------------------------
# Query planning integration
# ---------------------------------------------------------------------------

def test_retrieval_with_query_planner(vector_store):
    _seed(vector_store, [("Paris is in France.", "d1", "geo")])
    from semantic_vault.query_planner import QueryPlanner
    plan = QueryPlan("Where is Paris?", intent="factual", sub_questions=["Where is Paris?"],
                     strategies=["dense"])
    planner = QueryPlanner(gemini_client=_plan_client(plan))
    client = _synth_client()
    result = retrieve_and_synthesize(
        "Where is Paris?",
        vector_store,
        gemini_client=client,
        query_planner=planner,
    )
    assert result.answer == "Test answer."


def test_multi_sub_question_retrieval_fuses_results(vector_store):
    """Two sub-questions retrieve different chunks; results should be merged."""
    _seed(vector_store, [
        ("Alice works at Nexus AI.", "d1", "nexus"),
        ("Bob works at QuantumEdge.", "d2", "qedge"),
    ])
    from semantic_vault.query_planner import QueryPlanner
    plan = QueryPlan(
        "Who works at Nexus and QuantumEdge?",
        sub_questions=["Who works at Nexus AI?", "Who works at QuantumEdge?"],
        strategies=["dense"],
    )
    planner = QueryPlanner(gemini_client=_plan_client(plan))
    client = _synth_client()
    result = retrieve_and_synthesize(
        "Who works at Nexus and QuantumEdge?",
        vector_store,
        gemini_client=client,
        query_planner=planner,
    )
    assert result.sources  # both chunks should be retrieved


# ---------------------------------------------------------------------------
# RRF fusion correctness
# ---------------------------------------------------------------------------

def test_rrf_items_in_multiple_lists_rank_higher():
    shared = {"id": "shared", "text": "shared chunk", "score": 0.5}
    unique = {"id": "unique", "text": "unique chunk", "score": 0.9}
    list_a = [shared, unique]
    list_b = [shared]
    fused = rrf_fuse([list_a, list_b])
    assert fused[0]["id"] == "shared"


# ---------------------------------------------------------------------------
# LLM reranking integration
# ---------------------------------------------------------------------------

def test_retrieval_with_reranking_calls_gemini_twice(vector_store):
    """First call: synthesis; second call: reranker. Actually reranker fires first."""
    _seed(vector_store, [
        (f"Doc {i} content.", f"d{i}", f"doc{i}") for i in range(5)
    ])

    call_count = 0
    responses = [
        # reranker response
        type("R", (), {"text": json.dumps([3, 5, 1, 4, 2])})(),
        # synthesis response
        type("R", (), {"text": "Reranked answer."})(),
    ]

    def generate(*args, **kwargs):
        nonlocal call_count
        r = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return r

    client = MagicMock()
    client.models.generate_content.side_effect = generate

    # top_k=2 so that 5 stored chunks > 2 = top_k, triggering llm_rerank
    result = retrieve_and_synthesize(
        "relevant topic",
        vector_store,
        gemini_client=client,
        rerank=True,
        top_k=2,
    )
    assert client.models.generate_content.call_count >= 2


# ---------------------------------------------------------------------------
# Authority scorer integration
# ---------------------------------------------------------------------------

def test_authority_section_included_in_prompt(vector_store):
    _seed(vector_store, [("Test content.", "d1", "trusted_doc")])
    from semantic_vault.authority_scorer import AuthorityScorer
    scorer = MagicMock(spec=AuthorityScorer)
    scorer.format_for_prompt.return_value = "Source authority:\n  trusted_doc: trust=high (0.90)"

    client = _synth_client()
    retrieve_and_synthesize(
        "What is test content?",
        vector_store,
        gemini_client=client,
        authority_scorer=scorer,
    )
    scorer.format_for_prompt.assert_called_once()
    prompt_str = str(client.models.generate_content.call_args)
    assert "authority" in prompt_str.lower() or "trust" in prompt_str.lower()


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_phase3_params_all_optional(vector_store):
    """retrieve_and_synthesize must work with zero Phase 3 args (Phase 1 mode)."""
    _seed(vector_store, [("Content.", "d1", "doc")])
    client = _synth_client("Phase 1 answer.")
    result = retrieve_and_synthesize("query", vector_store, gemini_client=client)
    assert result.answer == "Phase 1 answer."
    assert result.graph_facts == []


def test_query_planner_none_uses_heuristic_classification(vector_store):
    """Without a planner, relational queries should still activate graph branch."""
    _seed(vector_store, [("Alice works with Bob.", "d1", "people")])
    gs = MagicMock()
    gs.text_to_cypher_query.return_value = ([], "")
    client = _synth_client()
    retrieve_and_synthesize(
        "Who does Alice work with?",
        vector_store,
        graph_store=gs,
        gemini_client=client,
    )
    gs.text_to_cypher_query.assert_called_once()
