"""LLM-powered entity + relationship extraction using Gemini."""
from __future__ import annotations

import time

from google import genai
from google.genai import types

from .config import settings
from .models import BASE_ENTITY_TYPES, ExtractionOutput

_EXTRACTION_PROMPT = """\
Extract structured knowledge from the text chunk below.

── ENTITIES ────────────────────────────────────────────────────────────────
Entity types to use (choose the closest match):
  Person, Organization, Location, Event, Concept, Document, Product, Date

For each entity:
  - name: the canonical / most-complete form of the name
  - type: exactly one of the types above
  - aliases: other names or spellings found in this chunk only
  - confidence: 0.0–1.0 reflecting how clearly the text establishes the entity

── RELATIONSHIPS ────────────────────────────────────────────────────────────
For each clear, direct relationship between two entities you extracted:
  - subject: the entity name (must be in your entities list)
  - predicate: ALL_CAPS_UNDERSCORED verb (e.g. WORKS_AT, FOUNDED, LOCATED_IN,
    COLLABORATED_WITH, IS_PART_OF, ACQUIRED, LEADS, AUTHORED)
  - object: the entity name (must be in your entities list)
  - confidence: 0.0–1.0
  - valid_from: ISO-8601 date string if a start date is stated, else empty string
  - valid_to: ISO-8601 date string if an end date is stated, else empty string

Only include relationships where both entities appear in the entities list.
Only include relationships that are explicitly stated, not inferred.

── OTHER FIELDS ────────────────────────────────────────────────────────────
  - key_facts: the 3–5 most important standalone factual statements (full sentences)
  - summary: 1–2 sentence description of what this chunk is about
  - topics: 3–10 keywords or themes

Text chunk:
{text}
"""


def _make_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def extract(text: str, client: genai.Client | None = None) -> ExtractionOutput:
    """Extract entities, relationships, and facts from *text* using Gemini.

    Retries once on transient errors with a short back-off.
    """
    if client is None:
        client = _make_client()

    prompt = _EXTRACTION_PROMPT.format(text=text)
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExtractionOutput,
    )

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=cfg,
            )
            result: ExtractionOutput = response.parsed  # type: ignore[assignment]
            if result is None:
                result = ExtractionOutput.model_validate_json(response.text)
            _normalise(result)
            return result
        except Exception as exc:
            if attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError(f"Extraction failed after retries: {exc}") from exc

    raise RuntimeError("Extraction failed")  # unreachable


def _normalise(result: ExtractionOutput) -> None:
    """Clamp entity types and normalise relationship predicates in-place."""
    entity_names = {e.name for e in result.entities}
    for entity in result.entities:
        if entity.type not in BASE_ENTITY_TYPES:
            entity.type = "Concept"

    # Keep only relationships whose both endpoints are in the entity list
    valid_rels = []
    for rel in result.relationships:
        if rel.subject in entity_names and rel.object in entity_names:
            rel.predicate = _normalise_predicate(rel.predicate)
            valid_rels.append(rel)
    result.relationships = valid_rels


def _normalise_predicate(pred: str) -> str:
    """'works at' → 'WORKS_AT', 'is-a' → 'IS_A'."""
    import re
    cleaned = re.sub(r"[^A-Z0-9_]+", "_", pred.upper().strip()).strip("_")
    return cleaned or "RELATES_TO"


# Expose for tests
_clamp_entity_types = _normalise  # backwards-compat alias
