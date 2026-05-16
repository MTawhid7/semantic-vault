"""Three-stage entity resolution pipeline.

Stage 1 — Embedding blocking  : ANN search in EntityIndex.
Stage 2 — LLM disambiguation  : Gemini decides for ambiguous pairs.
Stage 3 — Canonical management : Return existing ID or mint a new UUID.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable

from google import genai

from .config import settings
from .models import ExtractedEntity
from .storage import EntityIndex


@dataclass
class ResolutionResult:
    canonical_id: str
    is_new: bool                   # True → a new entity node must be created
    merged_with_name: str = ""     # name of the existing entity we merged with
    stage: str = ""                # "blocking_high" | "llm" | "new"
    confidence: float = 1.0


class EntityResolver:
    """Resolves a new ExtractedEntity to an existing canonical ID or creates one.

    Inject ``gemini_client`` to avoid re-creating a client per call.
    Inject ``_llm_fn`` to mock the LLM step in tests.
    """

    def __init__(
        self,
        entity_index: EntityIndex,
        *,
        gemini_client: genai.Client | None = None,
        high_threshold: float | None = None,
        low_threshold: float | None = None,
        _llm_fn: Callable[[str, str, str], bool] | None = None,
    ) -> None:
        self._index = entity_index
        self._client = gemini_client
        self._high = high_threshold if high_threshold is not None else settings.er_high_threshold
        self._low = low_threshold if low_threshold is not None else settings.er_low_threshold
        self._llm_fn = _llm_fn  # (name_a, name_b, context) → bool (True=same entity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, entity: ExtractedEntity, context: str = "") -> ResolutionResult:
        """Return a ResolutionResult for *entity* given the surrounding *context*."""
        candidates = self._index.search(
            name=entity.name,
            entity_type=entity.type,
            context=context,
            top_k=5,
        )

        if not candidates:
            return self._create_new(entity, context)

        top = candidates[0]

        # Stage 1 — definite match
        if top["score"] >= self._high:
            return ResolutionResult(
                canonical_id=top["canonical_id"],
                is_new=False,
                merged_with_name=top["name"],
                stage="blocking_high",
                confidence=top["score"],
            )

        # Stage 1 — definitely different
        if top["score"] < self._low:
            return self._create_new(entity, context)

        # Stage 2 — ambiguous: ask LLM
        return self._llm_disambiguate(entity, context, top)

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _create_new(self, entity: ExtractedEntity, context: str) -> ResolutionResult:
        canonical_id = str(uuid.uuid4())
        self._index.upsert(canonical_id, entity.name, entity.type, context)
        return ResolutionResult(
            canonical_id=canonical_id,
            is_new=True,
            stage="new",
            confidence=1.0,
        )

    def _llm_disambiguate(
        self,
        entity: ExtractedEntity,
        context: str,
        candidate: dict,
    ) -> ResolutionResult:
        """Ask Gemini whether *entity* and *candidate* refer to the same real-world entity."""
        if self._llm_fn is not None:
            same = self._llm_fn(entity.name, candidate["name"], context)
        else:
            same = self._call_gemini(entity.name, entity.type, candidate, context)

        if same:
            return ResolutionResult(
                canonical_id=candidate["canonical_id"],
                is_new=False,
                merged_with_name=candidate["name"],
                stage="llm",
                confidence=candidate["score"],
            )
        return self._create_new(entity, context)

    def _call_gemini(
        self,
        name_a: str,
        type_a: str,
        candidate: dict,
        context: str,
    ) -> bool:
        client = self._client or genai.Client(api_key=settings.gemini_api_key)
        prompt = f"""\
Do these two entity mentions refer to the SAME real-world entity?

Entity A: "{name_a}" (type: {type_a})
Entity B: "{candidate['name']}" (type: {candidate['type']})
Context where A appears: "{context[:400]}"

Answer with a single word: YES or NO."""
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        return response.text.strip().upper().startswith("YES")
