"""Neo4j knowledge graph store for Phase 2.

All graph operations are optional — if NEO4J_URI is not configured the
ingestion and retrieval pipelines fall back to Phase 1 behaviour.

Node model:
    (:Entity {canonical_id, name, type, aliases[]})

Edge model:
    (:Entity)-[:RELATES_TO {
        predicate, confidence,
        valid_from, valid_to,   # ISO-8601 strings or ""
        extracted_at,           # ingestion ISO timestamp
        source_chunk_id
    }]->(:Entity)

    (:Entity)-[:CONTRADICTS {reason}]->(:Entity)
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from google import genai

from .config import settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraphStore:
    """Wraps the Neo4j driver.  Instantiate only when NEO4J_URI is set."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        _driver: Any = None,  # injectable for tests
    ) -> None:
        if _driver is not None:
            self._driver = _driver
            return

        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Install the neo4j driver: pip install neo4j>=5.0"
            ) from exc

        self._driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def create_indexes(self) -> None:
        """Create uniqueness constraint + index.  Safe to call multiple times."""
        with self._driver.session() as s:
            s.run(
                "CREATE CONSTRAINT entity_canonical_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE"
            )
            s.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")

    # ------------------------------------------------------------------
    # Write — entities
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        canonical_id: str,
        name: str,
        entity_type: str,
        aliases: list[str] | None = None,
    ) -> None:
        """Create or update an Entity node (idempotent via MERGE)."""
        with self._driver.session() as s:
            s.run(
                """
                MERGE (e:Entity {canonical_id: $id})
                SET e.name = $name, e.type = $type
                WITH e, $aliases AS new_aliases
                CALL {
                    WITH e, new_aliases
                    UNWIND new_aliases AS alias
                    WITH e, alias
                    WHERE NOT alias IN coalesce(e.aliases, [])
                    SET e.aliases = coalesce(e.aliases, []) + [alias]
                }
                """,
                id=canonical_id,
                name=name,
                type=entity_type,
                aliases=list({name} | set(aliases or [])),
            )

    def add_alias(self, canonical_id: str, alias: str) -> None:
        """Append *alias* to an existing entity node (if not already present)."""
        with self._driver.session() as s:
            s.run(
                """
                MATCH (e:Entity {canonical_id: $id})
                WHERE NOT $alias IN coalesce(e.aliases, [])
                SET e.aliases = coalesce(e.aliases, []) + [$alias]
                """,
                id=canonical_id,
                alias=alias,
            )

    # ------------------------------------------------------------------
    # Write — relationships
    # ------------------------------------------------------------------

    def upsert_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        *,
        confidence: float = 1.0,
        valid_from: str = "",
        valid_to: str = "",
        source_chunk_id: str = "",
    ) -> None:
        """Create or refresh a RELATES_TO edge.  Detects conflicts automatically."""
        extracted_at = _now_iso()

        # Check for an existing contradicting fact (same subject+predicate, different object)
        with self._driver.session() as s:
            conflicting = s.run(
                """
                MATCH (s:Entity {canonical_id: $sid})
                      -[r:RELATES_TO {predicate: $pred}]->
                      (o:Entity)
                WHERE o.canonical_id <> $oid
                  AND r.valid_to = ""
                RETURN o.canonical_id AS oid, r.confidence AS conf
                LIMIT 1
                """,
                sid=subject_id,
                pred=predicate,
                oid=object_id,
            ).single()

        if conflicting:
            self._mark_conflict(subject_id, predicate, object_id, conflicting["oid"])

        with self._driver.session() as s:
            s.run(
                """
                MATCH (s:Entity {canonical_id: $sid})
                MATCH (o:Entity {canonical_id: $oid})
                MERGE (s)-[r:RELATES_TO {
                    predicate: $pred, source_chunk_id: $chunk_id
                }]->(o)
                SET r.confidence    = $conf,
                    r.valid_from    = $vf,
                    r.valid_to      = $vt,
                    r.extracted_at  = $ext
                """,
                sid=subject_id,
                oid=object_id,
                pred=predicate,
                chunk_id=source_chunk_id,
                conf=confidence,
                vf=valid_from,
                vt=valid_to,
                ext=extracted_at,
            )

    def _mark_conflict(
        self, subject_id: str, predicate: str, new_object_id: str, existing_object_id: str
    ) -> None:
        reason = f"Conflicting values for predicate {predicate}"
        with self._driver.session() as s:
            s.run(
                """
                MATCH (n:Entity {canonical_id: $nid})
                MATCH (e:Entity {canonical_id: $eid})
                MERGE (n)-[c:CONTRADICTS {predicate: $pred}]->(e)
                SET c.reason = $reason
                """,
                nid=new_object_id,
                eid=existing_object_id,
                pred=predicate,
                reason=reason,
            )

    # ------------------------------------------------------------------
    # Read — graph traversal
    # ------------------------------------------------------------------

    def traverse(self, entity_name: str, hops: int = 2) -> list[dict[str, Any]]:
        """Return relationships reachable from *entity_name* within *hops* steps."""
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (start:Entity)
                WHERE start.name = $name
                   OR $name IN coalesce(start.aliases, [])
                MATCH (start)-[r:RELATES_TO*1..$hops]->(related)
                RETURN start.name      AS source,
                       related.name    AS target,
                       related.type    AS target_type,
                       [rel IN r | rel.predicate] AS predicates,
                       [rel IN r | rel.confidence] AS confidences
                LIMIT 40
                """,
                name=entity_name,
                hops=hops,
            )
            return [dict(row) for row in result]

    def find_entity(self, name: str) -> dict[str, Any] | None:
        """Return the canonical entity node for *name* or one of its aliases."""
        with self._driver.session() as s:
            row = s.run(
                """
                MATCH (e:Entity)
                WHERE e.name = $name OR $name IN coalesce(e.aliases, [])
                RETURN e.canonical_id AS id, e.name AS name, e.type AS type,
                       e.aliases AS aliases
                LIMIT 1
                """,
                name=name,
            ).single()
            return dict(row) if row else None

    def get_entity_relationships(self, canonical_id: str) -> list[dict[str, Any]]:
        """Return all direct relationships for an entity."""
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (s:Entity {canonical_id: $id})-[r:RELATES_TO]->(o:Entity)
                RETURN o.name AS target, o.type AS target_type,
                       r.predicate AS predicate, r.confidence AS confidence,
                       r.valid_from AS valid_from, r.valid_to AS valid_to
                ORDER BY r.confidence DESC
                """,
                id=canonical_id,
            )
            return [dict(row) for row in result]

    # ------------------------------------------------------------------
    # Read — text-to-cypher
    # ------------------------------------------------------------------

    def text_to_cypher_query(
        self, question: str, gemini_client: genai.Client
    ) -> tuple[list[dict[str, Any]], str]:
        """Generate a Cypher query from *question* using Gemini, execute it, return results."""
        schema = (
            "Node: (:Entity {canonical_id, name, type, aliases[]})\n"
            "Edge: (:Entity)-[:RELATES_TO {predicate, confidence, valid_from, valid_to}]->(:Entity)\n"
            "Edge: (:Entity)-[:CONTRADICTS {predicate, reason}]->(:Entity)\n"
        )
        prompt = f"""\
Generate a read-only Neo4j Cypher query to answer the question about a knowledge graph.

Schema:
{schema}

Question: {question}

Rules:
- Use only MATCH and RETURN (no CREATE, MERGE, SET, DELETE)
- Include LIMIT 20
- Return entity names, types, and relationship predicates
- If the question is not about entity relationships, return exactly: MATCH (e:Entity) RETURN e LIMIT 0

Return ONLY the Cypher query, no explanation."""
        response = gemini_client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        cypher = re.sub(r"^```[a-z]*\n?", "", response.text.strip(), flags=re.IGNORECASE)
        cypher = re.sub(r"\n?```$", "", cypher).strip()

        try:
            with self._driver.session() as s:
                result = s.run(cypher)
                rows = [dict(row) for row in result]
            return rows, cypher
        except Exception:
            return [], cypher

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def entity_count(self) -> int:
        with self._driver.session() as s:
            row = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()
            return row["n"] if row else 0

    def relationship_count(self) -> int:
        with self._driver.session() as s:
            row = s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n").single()
            return row["n"] if row else 0
