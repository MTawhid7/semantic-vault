"""Retrieval pipeline: query → search → LLM synthesis.

Phase 1: vector search → synthesis.
Phase 2: graph traversal branch for relational queries.
Phase 3: query planning, parallel retrieval, RRF fusion, LLM reranking,
         source authority weighting.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from google import genai

from .config import settings
from .models import QueryResult
from .storage import VectorStore

_SYNTHESIS_PROMPT = """\
Answer the user's question using the retrieved knowledge below as your evidence.

Rules:
- Cite every factual claim inline as [Source: <doc_name>, chunk <N>].
- You may reason over the evidence and draw well-supported inferences, including indirect
  or multi-hop connections (e.g. A works with B, B is connected to C means A is indirectly
  linked to C). Label inferences clearly: "This implies..." or "Indirectly, ...".
- If two sources contradict each other, present both versions and flag the conflict.
- If the evidence is genuinely insufficient, say so and explain what is missing. Do not
  say "no connection exists" when evidence of an indirect connection is present — instead
  describe the indirect path you found.
- Be exhaustive: include all relevant names, numbers, dates, and roles from the evidence.

Retrieved chunks:
{chunks}
{graph_section}
{authority_section}
Question: {question}
"""

_GRAPH_SECTION_TEMPLATE = """\
Knowledge-graph facts (entity relationships):
{facts}
"""

# Heuristic keywords that suggest a relational query (Phase 2 graph branch)
_RELATIONAL_PATTERNS = re.compile(
    r"\b(who (does|did|is|are|was|were)|"
    r"related to|connected to|works? with|collaborat\w* with|"
    r"partners? of|colleagues? of|member of|belongs? to|"
    r"linked to|associated with|relationship between|"
    r"how is .+ related|which (company|organization|person|people)|"
    r"what (company|organization|group)|report(s|ed)? to|"
    r"founded by|acquired by|owned by)\b",
    re.IGNORECASE,
)


def _make_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def _is_relational_query(question: str) -> bool:
    return bool(_RELATIONAL_PATTERNS.search(question))


def _format_graph_rows(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = []
    for row in rows[:20]:
        lines.append("  • " + " | ".join(f"{k}: {v}" for k, v in row.items() if v is not None))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3: parallel retrieval + RRF fusion
# ---------------------------------------------------------------------------

def _retrieve_parallel(
    sub_questions: list[str],
    strategies: list[str],
    vector_store: VectorStore,
    graph_store: Any,
    client: genai.Client,
    top_k: int,
) -> tuple[list[list[dict]], list[dict]]:
    """Execute retrieval strategies in parallel for all sub-questions.

    Returns (vector_result_lists, graph_rows).
    """
    vector_results: dict[str, list[dict]] = {}
    graph_rows: list[dict] = []

    def do_dense(q: str) -> list[dict]:
        return vector_store.hybrid_search(q, top_k=top_k * 2)

    def do_graph(q: str) -> list[dict]:
        if graph_store is None:
            return []
        rows, _ = graph_store.text_to_cypher_query(q, client)
        return rows

    futures = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for q in sub_questions:
            if "dense" in strategies:
                futures[ex.submit(do_dense, q)] = ("dense", q)

        # Graph retrieval runs only for the first (original) sub-question.
        # Running Text2Cypher for every sub-question multiplies Gemini calls by N
        # and risks hitting rate limits or the parallel timeout on free-tier APIs.
        if "graph" in strategies and graph_store is not None and sub_questions:
            q0 = sub_questions[0]
            futures[ex.submit(do_graph, q0)] = ("graph", q0)

        for future in as_completed(futures, timeout=30):
            kind, q = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"  [warn] {kind} retrieval failed for '{q[:40]}': {exc}")
                result = []
            if kind == "dense":
                vector_results[q] = result
            else:
                graph_rows.extend(result)

    return list(vector_results.values()), graph_rows


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve_and_synthesize(
    query: str,
    vector_store: VectorStore,
    *,
    top_k: int | None = None,
    entity_type_filter: str | None = None,
    gemini_client: genai.Client | None = None,
    graph_store: Any = None,
    # Phase 3 optional components
    query_planner: Any = None,   # QueryPlanner | None
    authority_scorer: Any = None,  # AuthorityScorer | None
    rerank: bool | None = None,
) -> QueryResult:
    """Search the knowledge base and synthesise a grounded answer with Gemini.

    Phase 3 features activate when the corresponding component is supplied
    (query_planner, authority_scorer) or flag is True (rerank).
    """
    from .reranker import llm_rerank, rrf_fuse

    k = top_k if top_k is not None else settings.top_k
    client = gemini_client or _make_client()
    do_rerank = rerank if rerank is not None else settings.enable_reranking

    # ── Phase 3: query planning ────────────────────────────────────────────
    plan = None
    if query_planner is not None:
        plan = query_planner.plan(query)
        sub_questions = plan.sub_questions
        strategies = plan.strategies
        if plan.needs_reranking:
            do_rerank = True
    else:
        sub_questions = [query]
        strategies = ["dense"]
        if graph_store is not None and _is_relational_query(query):
            strategies.append("graph")

    # ── Parallel retrieval (Phase 3) or sequential (Phase 1/2) ────────────
    if len(sub_questions) > 1 or len(strategies) > 1:
        vector_result_lists, graph_rows = _retrieve_parallel(
            sub_questions, strategies, vector_store, graph_store, client, k
        )
        hits = rrf_fuse(vector_result_lists)[:k * 2]
    else:
        # When reranking, fetch extra candidates beyond top_k to give the reranker
        # something to work with — otherwise len(hits) == top_k → reranker skips.
        search_k = max(settings.rerank_top_n, k * 2) if do_rerank else k
        hits = vector_store.hybrid_search(
            query, top_k=search_k, entity_type_filter=entity_type_filter
        )
        graph_rows = []
        if graph_store is not None and _is_relational_query(query):
            try:
                graph_rows, _ = graph_store.text_to_cypher_query(query, client)
            except Exception as exc:
                print(f"  [warn] graph retrieval failed: {exc}")

    # ── Phase 3: LLM reranking ─────────────────────────────────────────────
    if do_rerank and len(hits) > k:
        hits = llm_rerank(query, hits, client, top_k=k, max_chunks_to_score=settings.rerank_top_n)
    else:
        hits = hits[:k]

    if not hits and not graph_rows:
        return QueryResult(
            answer="No relevant information found in the knowledge base.",
            sources=[],
            entities_mentioned=[],
            graph_facts=[],
        )

    # ── Build synthesis prompt ─────────────────────────────────────────────
    chunks_block = ""
    sources: list[dict] = []
    all_entity_names: list[str] = []
    doc_names_seen: list[str] = []

    for i, hit in enumerate(hits, start=1):
        doc_name = hit.get("source_doc_name", "unknown")
        chunk_idx = hit.get("chunk_index", "?")
        text = hit.get("text", "")
        chunks_block += f"\n[{i}] Source: {doc_name}, chunk {chunk_idx}\n{text}\n"
        sources.append(
            {
                "rank": i,
                "chunk_id": hit["id"],
                "doc_name": doc_name,
                "chunk_index": chunk_idx,
                "score": hit.get("score", 0.0),
                "snippet": text[:200] + ("..." if len(text) > 200 else ""),
                "topics": hit.get("topics", []),
            }
        )
        all_entity_names.extend(hit.get("entity_names", []))
        if doc_name not in doc_names_seen:
            doc_names_seen.append(doc_name)

    graph_section = ""
    if graph_rows:
        graph_section = _GRAPH_SECTION_TEMPLATE.format(facts=_format_graph_rows(graph_rows))

    # ── Phase 3: source authority section ─────────────────────────────────
    authority_section = ""
    if authority_scorer is not None and doc_names_seen:
        authority_section = authority_scorer.format_for_prompt(doc_names_seen)

    prompt = _SYNTHESIS_PROMPT.format(
        chunks=chunks_block or "(none)",
        graph_section=graph_section,
        authority_section=authority_section,
        question=query,
    )

    response = client.models.generate_content(
        model=settings.gemini_synthesis_model,
        contents=prompt,
    )

    return QueryResult(
        answer=response.text,
        sources=sources,
        entities_mentioned=list(dict.fromkeys(all_entity_names)),
        graph_facts=graph_rows,
    )
