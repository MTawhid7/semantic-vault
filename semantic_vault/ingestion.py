"""Ingestion pipeline: raw text → chunks → extraction → storage.

Phase 1: vector store + metadata store (always active).
Phase 2: entity resolution + knowledge graph (active when entity_index
         and graph_store are supplied).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from google import genai

from .chunker import chunk_text
from .config import settings
from .extractor import extract
from .models import Chunk, SourceDocument
from .storage import EntityIndex, MetadataStore, VectorStore


def ingest_text(
    content: str,
    name: str,
    vector_store: VectorStore,
    meta_store: MetadataStore,
    *,
    gemini_client: genai.Client | None = None,
    skip_extraction: bool = False,
    # Phase 2 — supply both or neither
    entity_index: EntityIndex | None = None,
    graph_store=None,  # GraphStore | None  (avoid circular import at module level)
    entity_resolver=None,  # EntityResolver | None
) -> SourceDocument:
    """Ingest *content* into the knowledge base.

    Raises ``ValueError`` if an identical document (same SHA-256) is already stored.
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    if meta_store.document_exists(content_hash):
        raise ValueError(f"Duplicate content detected for document '{name}'.")

    doc = SourceDocument(name=name, content_hash=content_hash)
    raw_chunks = chunk_text(content, settings.chunk_size_chars, settings.chunk_overlap_chars)

    chunks: list[Chunk] = []
    for idx, chunk_text_str in enumerate(raw_chunks):
        chunk = Chunk(
            text=chunk_text_str,
            source_doc_id=doc.id,
            source_doc_name=doc.name,
            chunk_index=idx,
        )

        if not skip_extraction:
            try:
                chunk.extraction = extract(chunk_text_str, client=gemini_client)
            except Exception as exc:
                print(f"  [warn] extraction failed for chunk {idx}: {exc}")

        # ── Phase 2: entity resolution + graph write ──────────────────────
        if entity_resolver is not None and graph_store is not None and chunk.extraction:
            _process_graph(chunk, entity_resolver, graph_store)

        chunks.append(chunk)

    doc.chunk_count = len(chunks)
    vector_store.upsert_chunks(chunks)
    meta_store.save_document(doc)
    return doc


def _process_graph(chunk: Chunk, entity_resolver, graph_store) -> None:
    """Resolve entities and write nodes + relationship edges to the graph."""
    ex = chunk.extraction
    if not ex:
        return

    # Resolve every entity → canonical_id
    canonical_ids: dict[str, str] = {}  # entity_name → canonical_id
    for entity in ex.entities:
        try:
            result = entity_resolver.resolve(entity, context=chunk.text)
            canonical_ids[entity.name] = result.canonical_id

            if result.is_new:
                graph_store.upsert_entity(
                    result.canonical_id,
                    entity.name,
                    entity.type,
                    aliases=entity.aliases,
                )
            else:
                # Add new surface form as alias on the existing canonical node
                graph_store.add_alias(result.canonical_id, entity.name)
                for alias in entity.aliases:
                    graph_store.add_alias(result.canonical_id, alias)

        except Exception as exc:
            print(f"  [warn] entity resolution failed for '{entity.name}': {exc}")

    # Write relationship edges (only when both endpoints are resolved)
    for rel in ex.relationships:
        subj_id = canonical_ids.get(rel.subject)
        obj_id = canonical_ids.get(rel.object)
        if subj_id and obj_id:
            try:
                graph_store.upsert_relationship(
                    subj_id,
                    rel.predicate,
                    obj_id,
                    confidence=rel.confidence,
                    valid_from=rel.valid_from,
                    valid_to=rel.valid_to,
                    source_chunk_id=chunk.id,
                )
            except Exception as exc:
                print(f"  [warn] graph relationship write failed: {exc}")
