"""Unit tests for the LLM extraction module (Gemini mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantic_vault.extractor import _clamp_entity_types, extract
from semantic_vault.models import BASE_ENTITY_TYPES, ExtractionOutput, ExtractedEntity


def _make_extraction(**kwargs) -> ExtractionOutput:
    defaults = dict(
        entities=[ExtractedEntity(name="Alice", type="Person", confidence=0.9)],
        key_facts=["Alice is a researcher."],
        summary="Alice is a researcher at MIT.",
        topics=["research", "academia"],
    )
    defaults.update(kwargs)
    return ExtractionOutput(**defaults)


def _mock_client(extraction: ExtractionOutput) -> MagicMock:
    response = MagicMock()
    response.parsed = extraction
    response.text = extraction.model_dump_json()
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------------


def test_extract_returns_extraction_output():
    expected = _make_extraction()
    client = _mock_client(expected)
    result = extract("Alice is a researcher at MIT.", client=client)
    assert isinstance(result, ExtractionOutput)
    assert result.summary == expected.summary


def test_extract_entities_populated():
    expected = _make_extraction()
    result = extract("text", client=_mock_client(expected))
    assert len(result.entities) == 1
    assert result.entities[0].name == "Alice"
    assert result.entities[0].type == "Person"


def test_extract_calls_generate_content_once():
    client = _mock_client(_make_extraction())
    extract("some text", client=client)
    client.models.generate_content.assert_called_once()


def test_clamp_entity_types_fixes_unknown():
    extraction = _make_extraction(
        entities=[
            ExtractedEntity(name="Aspirin", type="Drug", confidence=0.8),
            ExtractedEntity(name="London", type="Location", confidence=0.95),
        ]
    )
    _clamp_entity_types(extraction)
    types_found = {e.type for e in extraction.entities}
    assert "Drug" not in types_found
    assert "Concept" in types_found
    assert "Location" in types_found


def test_clamp_entity_types_leaves_valid_types_unchanged():
    extraction = _make_extraction(
        entities=[ExtractedEntity(name=t, type=t, confidence=0.9) for t in BASE_ENTITY_TYPES]
    )
    _clamp_entity_types(extraction)
    result_types = {e.type for e in extraction.entities}
    assert result_types == BASE_ENTITY_TYPES


def test_extract_retries_on_failure_then_succeeds():
    expected = _make_extraction()
    response_ok = MagicMock()
    response_ok.parsed = expected

    client = MagicMock()
    client.models.generate_content.side_effect = [RuntimeError("transient"), response_ok]

    # Should not raise despite first failure (retries once)
    result = extract("text", client=client)
    assert result.summary == expected.summary
    assert client.models.generate_content.call_count == 2


def test_extract_raises_after_two_failures():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("persistent error")
    with pytest.raises(RuntimeError, match="Extraction failed"):
        extract("text", client=client)


def test_extract_falls_back_to_text_when_parsed_is_none():
    expected = _make_extraction()
    response = MagicMock()
    response.parsed = None
    response.text = expected.model_dump_json()

    client = MagicMock()
    client.models.generate_content.return_value = response

    result = extract("text", client=client)
    assert result.summary == expected.summary
