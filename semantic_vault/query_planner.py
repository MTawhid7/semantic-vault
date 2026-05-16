"""Phase 3 — Query planner: decompose questions and select retrieval strategies.

The planner classifies the user's intent and splits complex multi-part questions
into focused sub-questions, each paired with the optimal retrieval strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

from .config import settings


# ---------------------------------------------------------------------------
# Plan model (also the Gemini structured-output schema)
# ---------------------------------------------------------------------------

class _GeminiQueryPlan(BaseModel):
    """Gemini output schema for query classification."""
    intent: str           # factual | relational | aggregation | temporal | comparison
    sub_questions: list[str]   # 1–3 focused sub-questions (include the original)
    strategies: list[str]      # subset of: dense, graph, structured
    hop_depth: int             # graph traversal depth: 1, 2, or 3
    needs_reranking: bool      # True for complex / multi-document questions


@dataclass
class QueryPlan:
    """Resolved query plan used by the retrieval pipeline."""
    original_query: str
    intent: str = "factual"
    sub_questions: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=lambda: ["dense"])
    hop_depth: int = 2
    needs_reranking: bool = False

    def __post_init__(self) -> None:
        if not self.sub_questions:
            self.sub_questions = [self.original_query]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """\
Analyse the user question below and produce a retrieval plan.

Intent options:
  factual      — asking for a specific fact, date, number, or definition
  relational   — asking about connections between entities (who, which, how linked)
  aggregation  — asking for a list or count of items
  temporal     — asking about events ordered in time
  comparison   — comparing two or more entities or facts

Strategy options (pick ALL that apply):
  dense      — semantic vector search (always include this)
  graph      — knowledge-graph traversal (include for relational/multi-hop questions)
  structured — exact field queries (include for temporal or exact-match questions)

Sub-questions: break the question into 1–3 focused sub-queries that together answer it.
  Always include the original question as one sub-question.

hop_depth: for graph traversal — 1 (direct), 2 (friend-of-friend), 3 (wider network).

needs_reranking: True when the question is complex, multi-document, or requires
  comparing evidence from multiple sources.

Question: {question}
"""

_VALID_INTENTS = frozenset(["factual", "relational", "aggregation", "temporal", "comparison"])
_VALID_STRATEGIES = frozenset(["dense", "graph", "structured"])


class QueryPlanner:
    """Classifies and decomposes a query into a retrieval plan using Gemini."""

    def __init__(self, gemini_client: genai.Client | None = None) -> None:
        self._client = gemini_client

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    def plan(self, question: str) -> QueryPlan:
        """Return a QueryPlan for *question*.  Never raises — falls back gracefully."""
        try:
            return self._call_gemini(question)
        except Exception as exc:
            print(f"  [warn] query planner failed ({exc}), using default plan")
            return QueryPlan(original_query=question)

    def _call_gemini(self, question: str) -> QueryPlan:
        client = self._get_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=_PLAN_PROMPT.format(question=question),
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_GeminiQueryPlan,
            ),
        )
        raw: _GeminiQueryPlan = response.parsed  # type: ignore[assignment]
        if raw is None:
            raw = _GeminiQueryPlan.model_validate_json(response.text)

        # Sanitise
        intent = raw.intent if raw.intent in _VALID_INTENTS else "factual"
        strategies = [s for s in raw.strategies if s in _VALID_STRATEGIES] or ["dense"]
        sub_questions = [q.strip() for q in raw.sub_questions if q.strip()][:3]
        if not sub_questions:
            sub_questions = [question]
        hop_depth = max(1, min(3, raw.hop_depth))

        return QueryPlan(
            original_query=question,
            intent=intent,
            sub_questions=sub_questions,
            strategies=strategies,
            hop_depth=hop_depth,
            needs_reranking=raw.needs_reranking,
        )
