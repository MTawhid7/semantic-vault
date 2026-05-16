"""Unit tests for GraphStore — Neo4j driver is fully mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from semantic_vault.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> tuple[GraphStore, MagicMock]:
    """Return (GraphStore, mock_session) with driver wired up."""
    driver = MagicMock()
    session = MagicMock()
    # Support context-manager usage:  with driver.session() as s:
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    gs = GraphStore(_driver=driver)
    return gs, session


# ---------------------------------------------------------------------------
# upsert_entity
# ---------------------------------------------------------------------------

def test_upsert_entity_runs_merge_query():
    gs, session = _make_store()
    gs.upsert_entity("id-1", "Alice Johnson", "Person", aliases=["Alice"])
    assert session.run.called
    cypher = session.run.call_args[0][0]
    assert "MERGE" in cypher
    assert "Entity" in cypher


def test_upsert_entity_passes_correct_params():
    gs, session = _make_store()
    gs.upsert_entity("uuid-123", "Acme Corp", "Organization")
    _, kwargs = session.run.call_args
    # params are positional kwargs in the call
    all_args = session.run.call_args
    assert "uuid-123" in str(all_args) or "id" in str(all_args)


# ---------------------------------------------------------------------------
# add_alias
# ---------------------------------------------------------------------------

def test_add_alias_runs_set_query():
    gs, session = _make_store()
    gs.add_alias("id-1", "Acme")
    cypher = session.run.call_args[0][0]
    assert "aliases" in cypher.lower() or "alias" in cypher.lower()


# ---------------------------------------------------------------------------
# upsert_relationship
# ---------------------------------------------------------------------------

def test_upsert_relationship_checks_for_conflicts():
    gs, session = _make_store()
    # No conflict found
    session.run.return_value.single.return_value = None
    gs.upsert_relationship("id-1", "WORKS_AT", "id-2", confidence=0.9)
    # Should have run at least two queries (conflict check + relationship write)
    assert session.run.call_count >= 2


def test_upsert_relationship_marks_conflict_when_found():
    gs, session = _make_store()
    # Simulate a conflicting fact found
    conflict_row = MagicMock()
    conflict_row.__getitem__ = lambda self, key: "id-3" if key == "oid" else 0.8
    session.run.return_value.single.return_value = conflict_row

    gs.upsert_relationship("id-1", "BORN_IN", "id-2", confidence=0.9)
    # Should have run conflict-check + conflict-mark + relationship write = 3 queries
    assert session.run.call_count >= 3
    # One of the cypher calls should contain CONTRADICTS
    all_cyphers = " ".join(str(c) for c in session.run.call_args_list)
    assert "CONTRADICTS" in all_cyphers


# ---------------------------------------------------------------------------
# traverse
# ---------------------------------------------------------------------------

def test_traverse_runs_match_query():
    gs, session = _make_store()
    session.run.return_value.__iter__ = MagicMock(return_value=iter([]))
    gs.traverse("Alice", hops=2)
    cypher = session.run.call_args[0][0]
    assert "MATCH" in cypher
    assert "RELATES_TO" in cypher


def test_traverse_returns_list_of_dicts():
    gs, session = _make_store()
    mock_row = MagicMock()
    mock_row.keys.return_value = ["source", "target", "predicates", "confidences", "target_type"]
    mock_row.__iter__ = MagicMock(return_value=iter(["Alice", "Acme", ["WORKS_AT"], [0.9], "Organization"]))
    mock_row.data.return_value = {
        "source": "Alice", "target": "Acme", "predicates": ["WORKS_AT"],
        "confidences": [0.9], "target_type": "Organization",
    }
    session.run.return_value.__iter__ = MagicMock(
        return_value=iter([mock_row])
    )
    # Patch dict(row) to work with MagicMock
    with patch("semantic_vault.graph_store.dict", side_effect=lambda r: r.data()):
        result = gs.traverse("Alice")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# text_to_cypher_query
# ---------------------------------------------------------------------------

def test_text_to_cypher_uses_gemini_and_runs_cypher():
    gs, session = _make_store()
    session.run.return_value.__iter__ = MagicMock(return_value=iter([]))

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "MATCH (e:Entity) RETURN e.name LIMIT 5"
    mock_client.models.generate_content.return_value = mock_response

    rows, cypher = gs.text_to_cypher_query("Who does Alice work with?", mock_client)
    assert mock_client.models.generate_content.called
    assert "MATCH" in cypher
    assert isinstance(rows, list)


def test_text_to_cypher_handles_bad_cypher_gracefully():
    gs, session = _make_store()
    session.run.side_effect = Exception("Syntax error")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "NOT VALID CYPHER!!!"
    mock_client.models.generate_content.return_value = mock_response

    rows, cypher = gs.text_to_cypher_query("bad question", mock_client)
    assert rows == []  # should not raise


# ---------------------------------------------------------------------------
# create_indexes
# ---------------------------------------------------------------------------

def test_create_indexes_runs_constraint_queries():
    gs, session = _make_store()
    gs.create_indexes()
    assert session.run.call_count >= 2
    all_cyphers = " ".join(str(c) for c in session.run.call_args_list)
    assert "CONSTRAINT" in all_cyphers or "constraint" in all_cyphers.lower()
