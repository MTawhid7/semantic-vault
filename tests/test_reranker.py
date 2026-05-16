"""Unit tests for rrf_fuse and llm_rerank — Gemini is mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from semantic_vault.reranker import llm_rerank, rrf_fuse


def _chunks(n: int, prefix: str = "chunk") -> list[dict]:
    return [{"id": f"id-{i}", "text": f"{prefix} {i}", "score": 1.0 / (i + 1)} for i in range(n)]


def _rerank_client(ratings: list[int]) -> MagicMock:
    response = MagicMock()
    response.text = json.dumps(ratings)
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------

def test_rrf_fuse_single_list():
    chunks = _chunks(5)
    fused = rrf_fuse([chunks])
    assert len(fused) == 5
    assert fused[0]["id"] == "id-0"  # highest rank in single list


def test_rrf_fuse_deduplicates_across_lists():
    list_a = _chunks(3)
    list_b = [list_a[0], list_a[1], {"id": "id-new", "text": "extra", "score": 0.5}]
    fused = rrf_fuse([list_a, list_b])
    ids = [r["id"] for r in fused]
    # id-0 and id-1 appear in both lists → higher RRF score → at top
    assert ids.count("id-0") == 1  # deduplicated
    assert "id-new" in ids


def test_rrf_fuse_promotes_items_in_multiple_lists():
    # item-shared appears in both lists → should rank above item-only-a
    list_a = [{"id": "shared", "text": "a", "score": 0.5}, {"id": "only-a", "text": "b", "score": 0.9}]
    list_b = [{"id": "shared", "text": "a", "score": 0.5}]
    fused = rrf_fuse([list_a, list_b])
    assert fused[0]["id"] == "shared"


def test_rrf_fuse_empty_lists():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_rrf_fuse_multiple_empty_and_one_full():
    chunks = _chunks(3)
    fused = rrf_fuse([[], chunks, []])
    assert len(fused) == 3


# ---------------------------------------------------------------------------
# llm_rerank
# ---------------------------------------------------------------------------

def test_llm_rerank_reorders_by_score():
    chunks = _chunks(6)
    # ratings: [1, 2, 5, 3, 4, 1] → chunk-2 (rating=5) should be first
    client = _rerank_client([1, 2, 5, 3, 4, 1])
    reranked = llm_rerank("query", chunks, client, top_k=5, max_chunks_to_score=6)
    assert reranked[0]["id"] == "id-2"


def test_llm_rerank_returns_top_k():
    chunks = _chunks(10)
    client = _rerank_client([5, 4, 3, 2, 1, 5, 4, 3, 2, 1])
    reranked = llm_rerank("query", chunks, client, top_k=3)
    assert len(reranked) == 3


def test_llm_rerank_falls_back_on_gemini_error():
    chunks = _chunks(5)
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("API down")
    reranked = llm_rerank("query", chunks, client, top_k=3)
    assert len(reranked) == 3  # returns original order truncated, no raise


def test_llm_rerank_skips_if_fewer_than_top_k():
    chunks = _chunks(3)
    client = MagicMock()
    reranked = llm_rerank("query", chunks, client, top_k=5)
    # fewer chunks than top_k → return as-is without calling Gemini
    client.models.generate_content.assert_not_called()
    assert len(reranked) == 3


def test_llm_rerank_handles_rating_length_mismatch():
    chunks = _chunks(5)
    client = _rerank_client([5, 3])  # wrong length → falls back
    reranked = llm_rerank("query", chunks, client, top_k=3)
    assert len(reranked) == 3  # no raise


def test_llm_rerank_strips_markdown_code_fences():
    chunks = _chunks(4)
    response = MagicMock()
    response.text = "```json\n[4, 3, 2, 1]\n```"
    client = MagicMock()
    client.models.generate_content.return_value = response
    reranked = llm_rerank("query", chunks, client, top_k=4)
    assert reranked[0]["id"] == "id-0"  # rated 4 → highest
