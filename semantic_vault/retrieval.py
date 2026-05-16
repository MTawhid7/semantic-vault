"""Retrieval pipeline: query → search → LLM synthesis.

Phase 1: vector search → synthesis.
Phase 2: adds a graph traversal branch for relational queries; results are
         merged before synthesis.
"""
from __future__ import annotations

import re

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
Question: {question}
"""

_GRAPH_SECTION_TEMPLATE = """\
Knowledge-graph facts (entity relationships):
{facts}
"""

# Heuristic keywords that suggest a relational query (graph branch)
_RELATIONAL_PATTERNS = re.compile(
    r"\b(who (does|did|is|are|was|were)|"
    r"related to|connected to|works? with|collaborated? with|"
    r"partners? of|colleagues? of|member of|belongs? to|"
    r"linked to|associated with|relationship between|"
    r"how is .+ related|which (company|organization|person|people)|"
    r"what (company|organization|group)|report(s|ed)? to|"
    r"founded by|acquired by|owned by|"
    r"collaborat\w* with)\b",
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
        # Try to render in a readable way regardless of exact column names
        lines.append("  • " + " | ".join(f"{k}: {v}" for k, v in row.items() if v is not None))
    return "\n".join(lines)


def retrieve_and_synthesize(
    query: str,
    vector_store: VectorStore,
    *,
    top_k: int | None = None,
    entity_type_filter: str | None = None,
    gemini_client: genai.Client | None = None,
    graph_store=None,  # GraphStore | None  (avoid circular import)
) -> QueryResult:
    """Search the knowledge base and synthesise a grounded answer with Gemini."""
    k = top_k if top_k is not None else settings.top_k
    client = gemini_client or _make_client()

    # ── Phase 1: vector search ────────────────────────────────────────────
    hits = vector_store.hybrid_search(query, top_k=k, entity_type_filter=entity_type_filter)

    # ── Phase 2: graph retrieval (when graph store available + relational Q) ─
    graph_rows: list[dict] = []
    cypher_used = ""
    if graph_store is not None and _is_relational_query(query):
        try:
            graph_rows, cypher_used = graph_store.text_to_cypher_query(query, client)
        except Exception as exc:
            print(f"  [warn] graph retrieval failed: {exc}")

    if not hits and not graph_rows:
        return QueryResult(
            answer="No relevant information found in the knowledge base.",
            sources=[],
            entities_mentioned=[],
            graph_facts=[],
        )

    # ── Build prompt context ──────────────────────────────────────────────
    chunks_block = ""
    sources: list[dict] = []
    all_entity_names: list[str] = []

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

    graph_section = ""
    if graph_rows:
        graph_section = _GRAPH_SECTION_TEMPLATE.format(facts=_format_graph_rows(graph_rows))

    prompt = _SYNTHESIS_PROMPT.format(
        chunks=chunks_block or "(none)",
        graph_section=graph_section,
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
