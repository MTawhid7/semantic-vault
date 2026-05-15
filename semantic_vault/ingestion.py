"""Ingestion pipeline: raw text → chunks → extraction → storage."""
from __future__ import annotations

import hashlib

from google import genai

from .chunker import chunk_text
from .config import settings
from .extractor import extract
from .models import Chunk, SourceDocument
from .storage import MetadataStore, VectorStore


def ingest_text(
    content: str,
    name: str,
    vector_store: VectorStore,
    meta_store: MetadataStore,
    *,
    gemini_client: genai.Client | None = None,
    skip_extraction: bool = False,
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
                # Extraction is best-effort; ingestion still succeeds without it
                print(f"  [warn] extraction failed for chunk {idx}: {exc}")
        chunks.append(chunk)

    doc.chunk_count = len(chunks)
    vector_store.upsert_chunks(chunks)
    meta_store.save_document(doc)
    return doc
