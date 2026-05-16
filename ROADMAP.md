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

### Capabilities
- Ingest any plaintext, notes, articles, research
- Automatic entity extraction (Person, Organization, Location, Event, Concept, Document, Product, Date)
- Relationship extraction (subject → predicate → object triples)
- Semantic search across all ingested content
- Cited answers grounded in stored knowledge
- Duplicate detection (SHA-256 content hash)
- Filter search results by entity type
- Persistent storage (qdrant_db/ + semantic_vault.db)
- Full test suite: 46 tests, no API key required for unit/integration tests

---

## ✅ Phase 2 — Knowledge Graph (complete)

**Goal:** Add explicit entity relationships and multi-hop reasoning.

### Stack additions
| Component | Technology |
|---|---|
| Graph database | Neo4j AuraDB Free / local Community |
| Graph driver | `neo4j` Python driver |
| Entity index | Qdrant `entity_vectors` collection (ANN blocking) |

### Pipeline
```
After extraction:
  → Entity Resolution (3-stage):
      1. Embedding blocking — ANN search in entity_vectors (≥0.85 cosine → merge)
      2. Gemini LLM disambiguation — for ambiguous pairs (0.65–0.85 range)
      3. Canonical management — one UUID per entity, aliases accumulated
  → Graph write (Neo4j):
      - MERGE entity nodes with name/type/aliases
      - CREATE RELATES_TO edges: predicate, confidence, valid_from/to, source_chunk_id
      - CONTRADICTS edges when conflicting facts detected

Query path:
  → _is_relational_query() heuristic classifier
  → Gemini Text2Cypher for relational queries → Neo4j execution
  → Vector results + graph facts merged in synthesis prompt
```

### New capabilities
- "Who works with Alice?" → graph traversal
- Multi-hop connection queries (1–3 hops)
- Entity deduplication: "Dr. Sarah Chen" / "S. Chen" → one canonical node
- Conflict tracking: two sources disagree → answer flags both with sources
- Provenance: every relationship traces to exact source chunk
- `rebuild-graph` CLI command for retroactive graph population
- Full extraction payload cached in Qdrant for fast graph rebuilds

### Test count: 90 (all passing, no API key required for unit/integration)

---

## ✅ Phase 3 — Adaptive System (complete)

**Goal:** Self-organising schema, agentic retrieval, source authority, and continuous refinement.

### Stack additions
| Component | Technology |
|---|---|
| Query planner | Gemini structured output (`QueryPlan`) |
| Parallel retrieval | `concurrent.futures.ThreadPoolExecutor` |
| Reranker | Gemini LLM-as-judge (scores chunks 1–5) |
| Authority scorer | SQLite `source_authority` table |
| Ontology engine | Cosine-threshold clustering + Gemini type naming |
| Schema registry | SQLite `schema_registry` table |

### Pipeline
```
Query
  → QueryPlanner: decompose into sub-questions, classify intent,
    select strategies (dense / graph / structured)
  → Parallel retrieval:
      Thread 1: dense vector search per sub-question
      Thread 2: graph traversal (if relational)
      Thread 3: structured filter (if temporal/exact)
  → RRF fusion across all vector result lists
  → LLM reranker: score top-20 chunks → return top-K
  → Synthesis with authority-weighted conflict resolution

Background (every N ingestions):
  → OntologyEngine clusters Concept entities by cosine similarity
  → Proposes new type names for large clusters (≥ threshold)
  → Updates schema registry with version bump
  → Queues retroactive re-extraction for affected documents
```

### New capabilities
- **Agentic query decomposition**: multi-part questions split and answered jointly
- **Parallel retrieval**: dense + graph + structured fire concurrently, not sequentially
- **RRF fusion**: Reciprocal Rank Fusion merges multiple result lists without score scaling
- **LLM reranking**: relevance-scored re-ordering beyond embedding cosine
- **Source authority**: corroboration-based trust scores weight conflict resolution
- **Ontology evolution**: Concept entities cluster into new typed ontology entries
- **Schema registry**: versioned type history, subsumption hierarchy, example entities

### Test count: 130+ (all passing)

---

## Design principles (all phases)

1. **Semantic-first, schema-second** — structure is derived from meaning, not predefined
2. **Provenance over certainty** — track every fact's origin; never delete, only invalidate
3. **Confidence-weighted everything** — entities, relationships, and facts carry scores
4. **Incremental over batch** — new data updates the graph locally, not globally
5. **Retrieval over recall** — answers are reconstructed from evidence, not stored text
