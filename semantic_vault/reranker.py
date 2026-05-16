"""Phase 3 — LLM-as-judge reranker.

Takes the top-N chunks from RRF fusion and asks Gemini to score their relevance
to the original query.  Returns the top-K by score.
"""
from __future__ import annotations

import json

from google import genai

from .config import settings

_RERANK_PROMPT = """\
You are a relevance judge. Rate each retrieved chunk's relevance to the query below on a
scale from 1 (irrelevant) to 5 (highly relevant and directly answers the query).

Return ONLY a JSON array of integers, one per chunk, in the same order.
Example for 4 chunks: [5, 2, 4, 1]

Query: {query}

Chunks:
{chunks_text}
"""


def rrf_fuse(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    Each list item must have an ``id`` key.  Items appearing in multiple lists
    accumulate score; the fused list is sorted descending by RRF score.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            item_id = str(item.get("id", id(item)))
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in items:
                items[item_id] = item

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [items[i] for i in ranked_ids]


def llm_rerank(
    query: str,
    chunks: list[dict],
    client: genai.Client,
    top_k: int = 8,
    max_chunks_to_score: int = 20,
) -> list[dict]:
    """Re-order *chunks* by LLM relevance score.

    Only the top *max_chunks_to_score* are sent to avoid large prompts.
    Falls back to the original order if Gemini fails.
    """
    if len(chunks) <= top_k:
        return chunks

    to_score = chunks[:max_chunks_to_score]

    chunks_text = "\n".join(
        f"[{i + 1}] {c.get('text', '')[:300].replace(chr(10), ' ')}"
        for i, c in enumerate(to_score)
    )
    prompt = _RERANK_PROMPT.format(query=query, chunks_text=chunks_text)

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        ratings: list[int] = json.loads(text)

        if len(ratings) != len(to_score):
            raise ValueError(f"Expected {len(to_score)} ratings, got {len(ratings)}")

        ranked = sorted(
            enumerate(to_score), key=lambda x: ratings[x[0]], reverse=True
        )
        reranked = [chunk for _, chunk in ranked]

        # Append any chunks beyond max_chunks_to_score unscored at the end
        reranked.extend(chunks[max_chunks_to_score:])
        return reranked[:top_k]

    except Exception as exc:
        print(f"  [warn] reranker failed ({exc}), using original order")
        return chunks[:top_k]
