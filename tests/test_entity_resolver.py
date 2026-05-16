"""Unit tests for EntityResolver — no Gemini API or Neo4j required."""
from __future__ import annotations

import pytest

from semantic_vault.entity_resolver import EntityResolver, ResolutionResult
from semantic_vault.models import ExtractedEntity


def _entity(name: str, etype: str = "Person") -> ExtractedEntity:
    return ExtractedEntity(name=name, type=etype, confidence=0.9)


# ---------------------------------------------------------------------------
# Stage 1: blocking — high threshold (definite match)
# ---------------------------------------------------------------------------

def test_identical_name_resolves_to_existing(entity_index):
    """Adding the same name twice should return the same canonical_id."""
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)
    r1 = resolver.resolve(_entity("Alice Johnson"), context="Alice works at Acme.")
    assert r1.is_new

    r2 = resolver.resolve(_entity("Alice Johnson"), context="Alice works at Acme.")
    assert not r2.is_new
    assert r2.canonical_id == r1.canonical_id
    assert r2.stage == "blocking_high"


def test_new_entity_creates_entry(entity_index):
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)
    r = resolver.resolve(_entity("Completely Unique Name XYZ"), context="some context")
    assert r.is_new
    assert r.stage == "new"


def test_different_entities_get_different_ids(entity_index):
    """Two clearly different entities must not be merged."""
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)
    r1 = resolver.resolve(_entity("Alice Johnson"), context="Alice is a biologist.")
    r2 = resolver.resolve(_entity("Bob Smith"), context="Bob is an engineer.")
    assert r1.canonical_id != r2.canonical_id


def test_entity_index_grows_after_new_entity(entity_index):
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)
    assert entity_index.count() == 0
    resolver.resolve(_entity("Entity One"), context="context")
    assert entity_index.count() == 1
    resolver.resolve(_entity("Entity Two"), context="context")
    assert entity_index.count() == 2


# ---------------------------------------------------------------------------
# Stage 2: LLM disambiguation
# ---------------------------------------------------------------------------

def test_llm_says_same_entity_merges(entity_index):
    """When LLM returns True, the ambiguous pair should resolve to same canonical_id."""
    resolver_first = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)
    r1 = resolver_first.resolve(_entity("J. Smith"), context="J. Smith works in finance.")

    # Create a second resolver that always says 'same entity' via LLM
    resolver_merge = EntityResolver(
        entity_index,
        high_threshold=0.99,  # no automatic merge
        low_threshold=-1.0,   # -1.0 ensures any cosine score falls in the ambiguous band
        _llm_fn=lambda a, b, c: True,
    )
    r2 = resolver_merge.resolve(_entity("John Smith"), context="John Smith works in finance.")
    assert r2.canonical_id == r1.canonical_id
    assert r2.stage == "llm"
    assert not r2.is_new


def test_llm_says_different_entity_creates_new(entity_index):
    resolver_first = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)
    r1 = resolver_first.resolve(_entity("J. Smith"), context="J. Smith works in finance.")

    resolver_different = EntityResolver(
        entity_index,
        high_threshold=0.99,
        low_threshold=-1.0,   # ensures ambiguous band, LLM decides
        _llm_fn=lambda a, b, c: False,  # always different
    )
    r2 = resolver_different.resolve(_entity("John Smith"), context="John Smith is a chef.")
    assert r2.canonical_id != r1.canonical_id
    assert r2.is_new


# ---------------------------------------------------------------------------
# Resolution result fields
# ---------------------------------------------------------------------------

def test_resolution_result_is_dataclass(entity_index):
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)
    r = resolver.resolve(_entity("Test Entity"), context="ctx")
    assert isinstance(r, ResolutionResult)
    assert r.canonical_id
    assert isinstance(r.is_new, bool)
    assert r.stage in ("blocking_high", "llm", "new")


def test_merged_result_has_merged_with_name(entity_index):
    resolver = EntityResolver(entity_index, _llm_fn=lambda a, b, c: False)
    r1 = resolver.resolve(_entity("Alice Johnson"), context="Alice works here.")
    r2 = resolver.resolve(_entity("Alice Johnson"), context="Alice works here.")
    assert r2.merged_with_name == "Alice Johnson"


# ---------------------------------------------------------------------------
# Multiple entity types don't interfere
# ---------------------------------------------------------------------------

def test_person_and_org_with_same_name_stay_separate(entity_index):
    """'Apple' as Concept and 'Apple' as Organization might conflict — same name,
    different type. Low threshold should keep them separate by default."""
    resolver = EntityResolver(
        entity_index,
        high_threshold=0.99,  # very high — only exact matches merge
        low_threshold=0.0,
        _llm_fn=lambda a, b, c: False,
    )
    r1 = resolver.resolve(_entity("Apple", "Organization"), context="Apple Inc makes iPhones.")
    r2 = resolver.resolve(_entity("Apple", "Concept"), context="An apple is a fruit.")
    # With LLM saying 'different', they must not merge
    assert r1.canonical_id != r2.canonical_id
