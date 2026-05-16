from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

BASE_ENTITY_TYPES = frozenset(
    ["Person", "Organization", "Location", "Event", "Concept", "Document", "Product", "Date"]
)


class ExtractedEntity(BaseModel):
    name: str
    type: str  # one of BASE_ENTITY_TYPES; validated post-extraction
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedRelationship(BaseModel):
    """A directed semantic relationship between two named entities."""
    subject: str    # entity name — must appear in entities[]
    predicate: str  # ALL_CAPS_UNDERSCORED verb, e.g. WORKS_AT, FOUNDED, LOCATED_IN
    object: str     # entity name — must appear in entities[]
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    valid_from: str = ""  # ISO-8601 date string if stated; otherwise empty
    valid_to: str = ""    # ISO-8601 date string if stated; otherwise empty


class ExtractionOutput(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    key_facts: list[str]
    summary: str
    topics: list[str]


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_doc_id: str
    source_doc_name: str
    chunk_index: int
    extraction: ExtractionOutput | None = None


class SourceDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    content_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chunk_count: int = 0


class QueryResult(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    entities_mentioned: list[str]
    graph_facts: list[dict[str, Any]] = Field(default_factory=list)  # Phase 2 graph hits
