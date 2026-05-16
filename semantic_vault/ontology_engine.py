"""Phase 3 — Ontology evolution engine.

Detects emerging entity-type clusters among 'Concept' entities, proposes
canonical type names via Gemini, and registers them in a schema_registry
SQLite table.  Retroactive re-extraction is queued as a job list for the
CLI / background worker to consume.

No external clustering library required — uses pure-numpy cosine-threshold
union-find clustering that works on the existing 3072-dim embeddings.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np
from google import genai

from .config import settings

# Minimum cluster size before a new type is proposed
MIN_CLUSTER_SIZE = 4
# Cosine similarity threshold for grouping Concept entities
CLUSTER_THRESHOLD = 0.75
# Max new types proposed per evolution cycle
MAX_NEW_TYPES_PER_CYCLE = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Schema registry (SQLite)
# ---------------------------------------------------------------------------

class SchemaRegistry:
    """Tracks the live ontology — base types + auto-detected types."""

    BASE_TYPES = [
        "Person", "Organization", "Location", "Event",
        "Concept", "Document", "Product", "Date",
    ]

    def __init__(self, db_path: str = "semantic_vault.db") -> None:
        self._path = db_path
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_registry (
                    type_name        TEXT PRIMARY KEY,
                    parent_type      TEXT NOT NULL DEFAULT 'Concept',
                    schema_version   INTEGER NOT NULL DEFAULT 1,
                    created_at       TEXT NOT NULL,
                    example_entities TEXT NOT NULL DEFAULT '[]',
                    status           TEXT NOT NULL DEFAULT 'active'
                )
                """
            )
            # Seed base types
            for t in self.BASE_TYPES:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO schema_registry
                        (type_name, parent_type, schema_version, created_at, status)
                    VALUES (?, 'root', 1, ?, 'active')
                    """,
                    (t, _now_iso()),
                )
            conn.commit()

    def register_type(
        self,
        type_name: str,
        parent_type: str = "Concept",
        example_entities: list[str] | None = None,
    ) -> bool:
        """Register a new auto-detected type.  Returns False if already exists."""
        import json
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT type_name FROM schema_registry WHERE type_name = ?", (type_name,)
            ).fetchone()
            if existing:
                return False
            version = (conn.execute(
                "SELECT MAX(schema_version) AS v FROM schema_registry"
            ).fetchone()["v"] or 1) + 1
            conn.execute(
                """
                INSERT INTO schema_registry
                    (type_name, parent_type, schema_version, created_at, example_entities, status)
                VALUES (?, ?, ?, ?, ?, 'active')
                """,
                (
                    type_name, parent_type, version, _now_iso(),
                    json.dumps(example_entities or []),
                ),
            )
            conn.commit()
        return True

    def list_types(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM schema_registry WHERE status = 'active' ORDER BY schema_version"
            ).fetchall()
            return [dict(r) for r in rows]

    def current_version(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(schema_version) AS v FROM schema_registry"
            ).fetchone()
            return row["v"] or 1


# ---------------------------------------------------------------------------
# Clustering helpers (pure numpy, no sklearn dependency)
# ---------------------------------------------------------------------------

def _union_find_cluster(
    embeddings: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Group indices whose pairwise cosine similarity >= threshold."""
    n = len(embeddings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-9)
    sim = normed @ normed.T  # (n, n) cosine similarity matrix

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Ontology engine
# ---------------------------------------------------------------------------

class OntologyEngine:
    """Detects emerging entity clusters and proposes new ontology types."""

    def __init__(
        self,
        entity_index,  # EntityIndex — avoids circular import
        schema_registry: SchemaRegistry,
        gemini_client: genai.Client | None = None,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
        cluster_threshold: float = CLUSTER_THRESHOLD,
    ) -> None:
        self._index = entity_index
        self._registry = schema_registry
        self._client = gemini_client
        self._min_size = min_cluster_size
        self._threshold = cluster_threshold

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_evolution(self) -> list[str]:
        """Run one evolution cycle.  Returns list of newly registered type names."""
        concept_points = self._get_concept_points()
        if len(concept_points) < self._min_size:
            return []

        names = [p["name"] for p in concept_points]
        embeddings = np.array([p["embedding"] for p in concept_points], dtype=np.float32)

        clusters = _union_find_cluster(embeddings, self._threshold)
        large_clusters = [c for c in clusters if len(c) >= self._min_size]

        new_types: list[str] = []
        for cluster_indices in large_clusters[:MAX_NEW_TYPES_PER_CYCLE]:
            member_names = [names[i] for i in cluster_indices]
            type_name = self._propose_type(member_names)
            if type_name and self._registry.register_type(
                type_name,
                parent_type="Concept",
                example_entities=member_names[:5],
            ):
                new_types.append(type_name)
                print(f"  [ontology] new type proposed: '{type_name}' ({len(member_names)} members)")

        return new_types

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_concept_points(self) -> list[dict]:
        """Retrieve all Concept entities with their embeddings from the entity index."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results: list[dict] = []
        offset = None
        while True:
            points, next_offset = self._index._qdrant.scroll(
                collection_name="entity_vectors",
                scroll_filter=Filter(
                    must=[FieldCondition(key="type", match=MatchValue(value="Concept"))]
                ),
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                vec = p.vector
                if isinstance(vec, dict):
                    vec = vec.get("dense", [])
                if vec:
                    results.append({
                        "canonical_id": p.payload.get("canonical_id", str(p.id)),
                        "name": p.payload.get("name", ""),
                        "embedding": vec,
                    })
            if next_offset is None:
                break
            offset = next_offset
        return results

    def _propose_type(self, member_names: list[str]) -> str | None:
        """Ask Gemini to propose a canonical type name for a cluster of entity names."""
        client = self._get_client()
        sample = member_names[:10]
        prompt = f"""\
The following entity names were clustered together because they are semantically similar.
Propose a short, precise ontology type name (2–3 words max, PascalCase, no spaces) that
best describes what category these entities belong to.

Entities: {', '.join(sample)}

Rules:
- Use PascalCase (e.g. ResearchInstitution, PharmaceuticalDrug)
- Must be more specific than 'Concept' but more general than any single entity
- Do NOT use: Person, Organization, Location, Event, Document, Product, Date
- Return ONLY the type name, nothing else."""
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            name = response.text.strip().split()[0]  # take first word only
            # Basic validation: PascalCase, no special chars
            import re
            if re.match(r"^[A-Z][a-zA-Z0-9]{2,30}$", name):
                return name
        except Exception as exc:
            print(f"  [warn] type proposal failed: {exc}")
        return None
