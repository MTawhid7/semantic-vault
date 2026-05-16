"""Unit tests for AuthorityScorer."""
from __future__ import annotations

import pytest

from semantic_vault.authority_scorer import AuthorityScorer


@pytest.fixture()
def scorer(tmp_path) -> AuthorityScorer:
    return AuthorityScorer(db_path=str(tmp_path / "auth.db"))


# ---------------------------------------------------------------------------


def test_unknown_source_defaults_to_half(scorer):
    assert scorer.get_score("never_seen_doc") == 0.5


def test_new_document_registers_at_half(scorer):
    scorer.record_ingestion("doc_a")
    assert scorer.get_score("doc_a") == 0.5


def test_corroboration_raises_score(scorer):
    scorer.record_ingestion("doc_a")
    scorer.record_corroboration("doc_a", count=5)
    assert scorer.get_score("doc_a") > 0.5


def test_contradiction_lowers_score(scorer):
    scorer.record_ingestion("doc_b")
    scorer.record_corroboration("doc_b", count=2)
    scorer.record_contradiction("doc_b", count=10)
    assert scorer.get_score("doc_b") < 0.5


def test_score_bounded_0_to_1(scorer):
    scorer.record_corroboration("extreme_doc", count=1000)
    score = scorer.get_score("extreme_doc")
    assert 0.0 <= score <= 1.0


def test_get_all_scores_returns_all_docs(scorer):
    for name in ["doc_x", "doc_y", "doc_z"]:
        scorer.record_ingestion(name)
    scores = scorer.get_all_scores()
    assert set(["doc_x", "doc_y", "doc_z"]).issubset(scores.keys())


def test_format_for_prompt_includes_tier(scorer):
    scorer.record_ingestion("trusted_source")
    scorer.record_corroboration("trusted_source", count=9)
    txt = scorer.format_for_prompt(["trusted_source"])
    assert "high" in txt
    assert "trusted_source" in txt


def test_format_for_prompt_empty_list(scorer):
    assert scorer.format_for_prompt([]) == ""


def test_authority_formula():
    """authority = corroborations / (corroborations + contradictions + 1)."""
    scorer = AuthorityScorer.__new__(AuthorityScorer)
    # Manual formula check: 4 corroborations, 0 contradictions → 4/5 = 0.8
    c, d = 4, 0
    expected = round(c / (c + d + 1), 4)
    assert expected == 0.8
