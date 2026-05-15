"""Retrieval pipeline: query → hybrid search → LLM synthesis."""
from __future__ import annotations

from google import genai

from .config import settings
from .models import QueryResult
from .storage import VectorStore

_SYNTHESIS_PROMPT = """\
Answer the user's question using ONLY the retrieved knowledge chunks below.
Be precise and factual. If the chunks contain insufficient information, say so explicitly.
Include inline citations in the format [Source: <doc_name>, chunk <N>].

Retrieved chunks:
{chunks}

Question: {question}
"""


def _make_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def retrieve_and_synthesize(
    query: str,
    vector_store: VectorStore,
    *,
    top_k: int | None = None,
    entity_type_filter: str | None = None,
    gemini_client: genai.Client | None = None,
) -> QueryResult:
    """Search the vector store and synthesise an answer with Gemini."""
    k = top_k if top_k is not None else settings.top_k
    client = gemini_client or _make_client()

    hits = vector_store.hybrid_search(query, top_k=k, entity_type_filter=entity_type_filter)

    if not hits:
        return QueryResult(
            answer="No relevant information found in the knowledge base.",
            sources=[],
            entities_mentioned=[],
        )

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

    prompt = _SYNTHESIS_PROMPT.format(chunks=chunks_block, question=query)
    response = client.models.generate_content(
        model=settings.gemini_synthesis_model,
        contents=prompt,
    )

    return QueryResult(
        answer=response.text,
        sources=sources,
        entities_mentioned=list(dict.fromkeys(all_entity_names)),  # deduplicated, ordered
    )
