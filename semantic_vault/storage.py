"""Dual-store backend: Qdrant (dense vectors) + SQLite (metadata).

Embeddings are generated with Gemini's embedding API — no native ONNX runtime
required. Pass ``embed_fn`` to the constructor to inject a mock for tests.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Callable

from google import genai
from google.genai import types as genai_types
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import settings
from .models import Chunk, SourceDocument

_COLLECTION = "chunks"
_ENTITY_COLLECTION = "entity_vectors"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """Qdrant-backed store with dense vector search powered by Gemini embeddings."""

    def __init__(
        self,
        url: str = ":memory:",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        """
        Args:
            url: Qdrant target. ``:memory:`` for in-process, a file path for
                 local persistence, or an ``http://...`` address for a server.
            embed_fn: Optional embedding function ``(texts) -> list[vector]``.
                      When provided, Gemini is not called — useful for tests.
        """
        if url == ":memory:":
            self._client = QdrantClient(":memory:")
        elif url.startswith("http"):
            self._client = QdrantClient(url=url)
        else:
            self._client = QdrantClient(path=url)

        self._embed_fn = embed_fn
        self._dim = settings.gemini_embed_dim
        # In-process cache: same text always yields the same vector within a session.
        # Eliminates duplicate Gemini calls when EntityIndex.search() and .upsert()
        # both embed the same "{name} ({type})" string for a new entity.
        self._embed_cache: dict[str, list[float]] = {}
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a dense embedding vector for each text, with in-process caching."""
        if self._embed_fn is not None:
            return self._embed_fn(texts)

        results: list[list[float] | None] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if text in self._embed_cache:
                results[i] = self._embed_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            new_vecs = self._gemini_embed(uncached_texts)
            for idx, text, vec in zip(uncached_indices, uncached_texts, new_vecs):
                self._embed_cache[text] = vec
                results[idx] = vec

        return results  # type: ignore[return-value]

    def _gemini_embed(self, texts: list[str]) -> list[list[float]]:
        client = genai.Client(api_key=settings.gemini_api_key)
        results: list[list[float]] = []
        for text in texts:
            for attempt in range(3):
                try:
                    response = client.models.embed_content(
                        model=settings.gemini_embed_model,
                        contents=[text],
                        config=genai_types.EmbedContentConfig(
                            output_dimensionality=settings.gemini_embed_dim,
                        ),
                    )
                    results.append(list(response.embeddings[0].values))
                    break
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(1.5 ** attempt)
                    else:
                        raise RuntimeError(f"Embedding failed: {exc}") from exc
        return results

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if _COLLECTION in existing:
            info = self._client.get_collection(_COLLECTION)
            stored_dim = info.config.params.vectors["dense"].size
            if stored_dim != self._dim:
                raise RuntimeError(
                    f"Existing Qdrant collection has {stored_dim}-dim vectors but "
                    f"GEMINI_EMBED_DIM={self._dim}. Delete '{settings.qdrant_url}' "
                    "to reset the vector store."
                )
            return

        self._client.create_collection(
            collection_name=_COLLECTION,
            vectors_config={"dense": VectorParams(size=self._dim, distance=Distance.COSINE)},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self.embed(texts)

        points: list[PointStruct] = []
        for chunk, vec in zip(chunks, vectors):
            ex = chunk.extraction
            payload: dict[str, Any] = {
                "text": chunk.text,
                "source_doc_id": chunk.source_doc_id,
                "source_doc_name": chunk.source_doc_name,
                "chunk_index": chunk.chunk_index,
                "summary": ex.summary if ex else "",
                "topics": ex.topics if ex else [],
                "key_facts": ex.key_facts if ex else [],
                # Flat lists used for Qdrant payload filtering
                "entity_names": [e.name for e in ex.entities] if ex else [],
                "entity_types": list({e.type for e in ex.entities}) if ex else [],
                # Full extraction data stored so graph can be rebuilt without re-extraction
                "entities": [
                    {"name": e.name, "type": e.type,
                     "aliases": e.aliases, "confidence": e.confidence}
                    for e in ex.entities
                ] if ex else [],
                "relationships": [
                    {"subject": r.subject, "predicate": r.predicate, "object": r.object,
                     "confidence": r.confidence,
                     "valid_from": r.valid_from, "valid_to": r.valid_to}
                    for r in ex.relationships
                ] if ex else [],
            }
            points.append(PointStruct(id=chunk.id, vector={"dense": vec}, payload=payload))

        self._client.upsert(collection_name=_COLLECTION, points=points)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        entity_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to *top_k* chunks ranked by cosine similarity."""
        query_vec = self.embed([query])[0]

        filt: Filter | None = None
        if entity_type_filter:
            filt = Filter(
                must=[FieldCondition(key="entity_types", match=MatchValue(value=entity_type_filter))]
            )

        response = self._client.query_points(
            collection_name=_COLLECTION,
            query=query_vec,
            using="dense",
            limit=top_k,
            query_filter=filt,
            with_payload=True,
        )
        return [{"id": str(r.id), "score": r.score, **r.payload} for r in response.points]

    def get_chunks_by_doc_id(self, source_doc_id: str) -> list[dict[str, Any]]:
        """Return all chunk payloads for a given document, ordered by chunk_index."""
        results: list[dict[str, Any]] = []
        offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source_doc_id", match=MatchValue(value=source_doc_id))]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            results.extend({"id": str(p.id), **p.payload} for p in points)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(results, key=lambda c: c.get("chunk_index", 0))

    def count(self) -> int:
        return self._client.count(collection_name=_COLLECTION).count


# ---------------------------------------------------------------------------
# Entity index — separate Qdrant collection for entity-resolution blocking
# ---------------------------------------------------------------------------


class EntityIndex:
    """Stores one embedding per canonical entity for ANN-based resolution blocking.

    Shares the Qdrant client and embed function of a parent VectorStore so that
    no extra connections or API clients are needed.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._qdrant = vector_store._client
        self._embed = vector_store.embed
        self._dim = vector_store._dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if _ENTITY_COLLECTION not in existing:
            self._qdrant.create_collection(
                collection_name=_ENTITY_COLLECTION,
                vectors_config={"dense": VectorParams(size=self._dim, distance=Distance.COSINE)},
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, canonical_id: str, name: str, entity_type: str, context: str = "") -> None:
        """Store or refresh the embedding for a canonical entity.

        Only name + type are embedded (not context) so that the same entity
        always produces the same vector regardless of which document it appears in.
        Context is used by the LLM disambiguation stage, not the blocking stage.
        """
        query_text = f"{name} ({entity_type})"
        vec = self._embed([query_text])[0]
        self._qdrant.upsert(
            collection_name=_ENTITY_COLLECTION,
            points=[
                PointStruct(
                    id=canonical_id,
                    vector={"dense": vec},
                    payload={"canonical_id": canonical_id, "name": name, "type": entity_type},
                )
            ],
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, name: str, entity_type: str = "", context: str = "", top_k: int = 5) -> list[dict[str, Any]]:
        """Return up to *top_k* existing entities ranked by similarity."""
        query_text = f"{name} ({entity_type})" if entity_type else name
        vec = self._embed([query_text])[0]
        response = self._qdrant.query_points(
            collection_name=_ENTITY_COLLECTION,
            query=vec,
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "canonical_id": r.payload["canonical_id"],
                "name": r.payload["name"],
                "type": r.payload.get("type", ""),
                "score": r.score,
            }
            for r in response.points
        ]

    def count(self) -> int:
        return self._qdrant.count(collection_name=_ENTITY_COLLECTION).count


# ---------------------------------------------------------------------------
# Metadata store (SQLite)
# ---------------------------------------------------------------------------


class MetadataStore:
    """SQLite-backed store for source document metadata."""

    def __init__(self, db_path: str = "semantic_vault.db") -> None:
        self._path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_documents (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    created_at  TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()

    def document_exists(self, content_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM source_documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            return row is not None

    def save_document(self, doc: SourceDocument) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO source_documents
                    (id, name, content_hash, created_at, chunk_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc.id, doc.name, doc.content_hash, doc.created_at.isoformat(), doc.chunk_count),
            )
            conn.commit()

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM source_documents ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM source_documents").fetchone()[0]
