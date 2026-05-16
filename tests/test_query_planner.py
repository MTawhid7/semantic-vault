"""Unit tests for QueryPlanner — Gemini is mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from semantic_vault.query_planner import QueryPlan, QueryPlanner, _GeminiQueryPlan


def _mock_client(plan_dict: dict) -> MagicMock:
    schema_obj = _GeminiQueryPlan(**plan_dict)
    response = MagicMock()
    response.parsed = schema_obj
    response.text = schema_obj.model_dump_json()
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def _make_plan(**kwargs) -> dict:
    defaults = dict(
        intent="factual",
        sub_questions=["What does TrainFlow do?"],
        strategies=["dense"],
        hop_depth=1,
        needs_reranking=False,
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------


def test_plan_returns_query_plan():
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan()))
    result = planner.plan("What does TrainFlow do?")
    assert isinstance(result, QueryPlan)


def test_plan_preserves_original_query():
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan()))
    result = planner.plan("Original question?")
    assert result.original_query == "Original question?"


def test_factual_query_uses_dense_only():
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan(
        intent="factual", strategies=["dense"]
    )))
    plan = planner.plan("When was Nexus AI founded?")
    assert "dense" in plan.strategies
    assert "graph" not in plan.strategies


def test_relational_query_includes_graph():
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan(
        intent="relational", strategies=["dense", "graph"], hop_depth=2
    )))
    plan = planner.plan("Who does Priya Nair report to?")
    assert "graph" in plan.strategies
    assert plan.hop_depth == 2


def test_sub_questions_populated():
    sub_qs = ["What is A?", "What is B?"]
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan(sub_questions=sub_qs)))
    plan = planner.plan("What is A and B?")
    assert len(plan.sub_questions) == 2


def test_invalid_intent_clamped_to_factual():
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan(intent="unknown_intent")))
    plan = planner.plan("something")
    assert plan.intent == "factual"


def test_invalid_strategies_filtered():
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan(
        strategies=["dense", "invalid_strategy", "graph"]
    )))
    plan = planner.plan("q")
    assert "invalid_strategy" not in plan.strategies
    assert "dense" in plan.strategies


def test_hop_depth_clamped_to_1_3():
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan(hop_depth=99)))
    plan = planner.plan("q")
    assert plan.hop_depth <= 3

    planner2 = QueryPlanner(gemini_client=_mock_client(_make_plan(hop_depth=-5)))
    plan2 = planner2.plan("q")
    assert plan2.hop_depth >= 1


def test_empty_sub_questions_defaults_to_original():
    planner = QueryPlanner(gemini_client=_mock_client(_make_plan(sub_questions=[])))
    plan = planner.plan("My question?")
    assert plan.sub_questions == ["My question?"]


def test_planner_falls_back_on_gemini_error():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("API down")
    planner = QueryPlanner(gemini_client=client)
    plan = planner.plan("What is X?")  # must not raise
    assert isinstance(plan, QueryPlan)
    assert plan.original_query == "What is X?"
    assert plan.strategies == ["dense"]
