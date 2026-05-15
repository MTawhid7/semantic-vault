"""LLM-powered entity and fact extraction using Gemini."""
from __future__ import annotations

import time

from google import genai
from google.genai import types

from .config import settings
from .models import BASE_ENTITY_TYPES, ExtractionOutput

_EXTRACTION_PROMPT = """\
Extract structured knowledge from the text chunk below.

Entity types to use (choose the closest match):
  Person, Organization, Location, Event, Concept, Document, Product, Date

Rules:
- name: the canonical/most-complete form of the entity name
- type: exactly one of the types listed above
- aliases: other names/spellings for the same entity mentioned in this chunk
- confidence: 0.0–1.0 reflecting how clearly the text establishes this entity
- key_facts: the 3–5 most important standalone factual statements (full sentences)
- summary: 1–2 sentence description of what this chunk is about
- topics: 3–10 keywords or themes from the chunk

Text chunk:
{text}
"""


def _make_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def extract(text: str, client: genai.Client | None = None) -> ExtractionOutput:
    """Extract entities and facts from *text* using a Gemini model.

    Retries once on transient errors with a short back-off.
    """
    if client is None:
        client = _make_client()

    prompt = _EXTRACTION_PROMPT.format(text=text)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExtractionOutput,
    )

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=config,
            )
            result: ExtractionOutput = response.parsed  # type: ignore[assignment]
            if result is None:
                # Fallback: parse text manually
                import json
                result = ExtractionOutput.model_validate_json(response.text)
            _clamp_entity_types(result)
            return result
        except Exception as exc:
            if attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError(f"Extraction failed after retries: {exc}") from exc

    raise RuntimeError("Extraction failed")  # unreachable but satisfies type checkers


def _clamp_entity_types(result: ExtractionOutput) -> None:
    """Normalise entity types to the base ontology; default to 'Concept'."""
    for entity in result.entities:
        if entity.type not in BASE_ENTITY_TYPES:
            entity.type = "Concept"
