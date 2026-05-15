# Roadmap

Semantic Vault is being built in three phases, each independently shippable. Each phase adds capability without breaking what came before.

---

## ✅ Phase 1 — Semantic Vault (complete)

**Goal:** Prove the core concept. Add unstructured text; ask questions; get cited answers.

### Stack
| Component | Technology |
|---|---|
| LLM (extraction + synthesis) | Gemini 3 Flash Preview (`google-genai`) |
| Embeddings | Gemini Embedding 2 — 3072 dims |
| Vector store | Qdrant (local file or in-memory) |
| Metadata store | SQLite |
| Web UI | Gradio 6 (two-panel split layout) |
| CLI | Python argparse |

### Pipeline

```
Raw text
  → sentence-boundary chunker (overlap)
  → Gemini: extract entities / facts / summary / topics (structured JSON)
  → Gemini: embed chunk → 3072-dim dense vector
  → Qdrant: store point (vector + full metadata payload)
  → SQLite: store document record

Query
  → Gemini: embed query
  → Qdrant: cosine similarity top-K
  → Gemini: synthesise answer with inline citations
```

### Capabilities
- Ingest any plaintext, notes, articles, research
- Automatic entity extraction (Person, Organization, Location, Event, Concept, Document, Product, Date)
- Semantic search across all ingested content
- Cited answers grounded in stored knowledge
- Duplicate detection (SHA-256 content hash)
- Filter search results by entity type
- Persistent storage (qdrant_db/ + semantic_vault.db)
- Full test suite: 46 tests, no API key required for unit/integration tests

### Known limitations (resolved in Phase 2)
- No explicit relationships between entities — only similarity-based retrieval
- No entity deduplication ("Alice Johnson" and "Alice" are separate entities)
- No multi-hop reasoning ("Who are Alice's colleagues?" requires graph traversal)
- Schema is fixed to the base 8 entity types

---

## 🔲 Phase 2 — Knowledge Graph

**Goal:** Add explicit entity relationships and multi-hop reasoning. Answer questions like *"Who does Alice collaborate with?"* and *"Which organisations are related to Project X?"*

### New stack additions
| Component | Technology |
|---|---|
| Graph database | Neo4j (local or AuraDB free tier) |
| Graph driver | `neo4j` Python driver |

### New pipeline steps

```
After extraction:
  → Entity Resolution (3-stage):
      1. Embedding similarity blocking (≥ 0.85 cosine → candidate merge)
      2. Gemini disambiguation for ambiguous pairs (0.65–0.85 range)
      3. Canonical node management (one UUID per entity, aliases stored)
  → Graph write (Neo4j):
      - Create/merge entity nodes with properties
      - Create typed relationship edges with confidence + provenance
      - Store source_chunk_id, extracted_at, confidence on each edge

Retrieval additions:
  → Query decomposition: detect relationship questions
  → Graph traversal (Cypher, 1–3 hops) for relationship-aware answers
  → Result fusion: combine vector hits + graph hits before synthesis
```

### Data model

```
(:Person {name, aliases[], canonical_id})
  -[:WORKS_AT {confidence, since, source_chunk_id}]->
(:Organization {name, type, canonical_id})

(:Chunk {id}) -[:MENTIONS]-> (:Entity)
(:SourceDocument {id}) -[:CONTAINS]-> (:Chunk)
```

### Bi-temporal provenance
Every fact edge carries:
- `valid_from`, `valid_to` — when the fact was true in the world
- `extracted_at` — when we learned it
- `confidence` — extraction confidence score
- `source_chunk_id` — which chunk is the evidence

Conflicting facts are never deleted — a `CONTRADICTS` edge links them. Resolution happens at query time (recency + confidence weighting).

### New capabilities
- "Who works with Alice?" → graph traversal
- "What organisations is this concept related to?" → multi-hop
- Entity deduplication: "Alice Johnson", "Alice", "A. Johnson" → one canonical node
- Conflict tracking: two sources disagree → answer flags the contradiction
- Provenance: every answer traces back to exact source chunks

---

## 🔲 Phase 3 — Adaptive System

**Goal:** Self-organising schema, agentic retrieval, and continuous refinement. The system grows smarter as more data is added.

### New stack additions
| Component | Technology |
|---|---|
| Schema registry | PostgreSQL (replaces SQLite) |
| Async job queue | Redis Streams or Celery |
| Reranker | ColBERT via `ragatouille` or LLM-as-judge |

### Ontology evolution

```
Background worker (every N documents):
  → Cluster entity embeddings with Leiden algorithm
  → Detect emerging clusters not covered by existing types
  → Propose new type name (Gemini-generated from cluster members)
  → Queue retroactive re-extraction for affected documents
  → Add type to schema registry with version bump
```

Type explosion is prevented by:
- Hard budget: max N types per depth level
- Type subsumption hierarchy (`Drug → Compound → Substance`)
- Confidence threshold before a new type is promoted

### Agentic retrieval

```
User query
  → Planner agent: decompose into sub-questions, classify intent
  → Retriever agents (parallel):
      - Dense vector search (Qdrant)
      - Graph traversal (Neo4j Cypher)
      - Structured filter (PostgreSQL — dates, entity names)
  → RRF fusion + ColBERT reranking
  → Synthesiser agent: generate answer with inline citations
```

### Source authority scoring
Every document source gets a trust score based on:
- How often its facts are corroborated by other sources
- Whether its facts have been contradicted
- Recency of the information

### Conflict resolution pipeline
```
Conflicting facts detected at ingest:
  → Store both with CONTRADICTS edge
  → Score by: source authority × recency × confidence
  → At query time: synthesiser presents both versions if unresolvable
  → Background job: flag high-confidence contradictions for user review
```

### New capabilities
- New knowledge domains auto-recognised and typed
- Retroactive re-extraction when schema evolves (without full reprocessing)
- Multi-agent query decomposition for complex questions
- Conflict-aware answers that cite disagreements
- Source credibility model
- p50 query latency target: < 2s (Phase 1: immediate); < 5s (Phase 3: agentic)

---

## Design principles (all phases)

1. **Semantic-first, schema-second** — structure is derived from meaning, not predefined
2. **Provenance over certainty** — track every fact's origin; never delete, only invalidate
3. **Confidence-weighted everything** — entities, relationships, and facts carry scores
4. **Incremental over batch** — new data updates the graph locally, not globally
5. **Retrieval over recall** — answers are reconstructed from evidence, not stored text
