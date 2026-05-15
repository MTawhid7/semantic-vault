# Architecture — Semantic Vault

> A full-phase technical reference covering working mechanisms, design decisions, and technology comparisons.

---

## Table of Contents

1. [System Intent](#1-system-intent)
2. [Core Design Principles](#2-core-design-principles)
3. [Data Models](#3-data-models)
4. [Phase 1 — Semantic Vault](#4-phase-1--semantic-vault)
5. [Phase 2 — Knowledge Graph](#5-phase-2--knowledge-graph)
6. [Phase 3 — Adaptive System](#6-phase-3--adaptive-system)
7. [Technology Decisions](#7-technology-decisions)
8. [Cross-Cutting Concerns](#8-cross-cutting-concerns)
9. [Performance Characteristics](#9-performance-characteristics)

---

## 1. System Intent

Traditional databases require a user to design a schema before storing data. Semantic Vault inverts this contract: the user provides raw, unstructured text, and the system automatically extracts structure, organises it, and reconstructs answers on demand.

The abstraction goal is to replace three roles with an LLM pipeline:

| Traditional Role | Replaced By |
|---|---|
| Database architect (schema design) | LLM extraction with adaptive ontology |
| Search engineer (indexing, ranking) | Embedding model + vector database |
| Data analyst (query writing) | Natural language retrieval agent |

The system is not a chatbot. It has no general world knowledge during retrieval — it answers exclusively from what has been ingested. This makes it precise, auditable, and grounded.

---

## 2. Core Design Principles

These five principles govern every architectural decision across all phases.

### 2.1 Semantic-first, schema-second
Structure is derived from meaning after the fact, not imposed before data arrives. The user never touches a schema. The LLM discovers what structure the data implies.

*Consequence:* Entities and relationships are extracted per-chunk, not predefined in a migration file. The ontology is a live artefact that evolves as the corpus grows.

### 2.2 Provenance over certainty
Every fact is stored with its origin. Facts are never deleted — only invalidated with a `valid_to` timestamp and a reason. When two sources contradict each other, both are retained.

*Consequence:* The storage layer is append-only by design. Queries must resolve conflicts at read time rather than at write time. This is more expensive at query time but guarantees no information is silently lost.

### 2.3 Confidence-weighted everything
Entities, relationships, facts, and retrieval results all carry explicit confidence scores. These scores propagate through the pipeline and influence synthesis.

*Consequence:* Nothing is binary (known/unknown). The system expresses calibrated uncertainty in its answers rather than hallucinating certainty.

### 2.4 Incremental over batch
New data updates only the affected region of the knowledge graph. There is no full-corpus reprocessing on ingestion. Schema evolution triggers targeted retroactive re-extraction, not a global rebuild.

*Consequence:* Ingestion latency is bounded per-document. The system is usable from the first document onwards; it does not require a minimum corpus size.

### 2.5 Retrieval over recall
The system does not memorise answers. It stores evidence and reconstructs answers from that evidence at query time. Every answer is synthesised fresh from retrieved chunks.

*Consequence:* Answers can change as new contradicting evidence is added. The system does not commit to stale answers. This trades predictability for accuracy.

---

## 3. Data Models

These models are stable across all phases. Later phases add fields but do not break Phase 1 contracts.

### 3.1 Chunk (the atomic unit)

```python
class ExtractedEntity(BaseModel):
    name: str           # canonical form: "Guido van Rossum"
    type: str           # one of BASE_ENTITY_TYPES
    aliases: list[str]  # ["Guido", "GvR"] — other forms found in this chunk
    confidence: float   # 0.0–1.0

class ExtractionOutput(BaseModel):
    entities:  list[ExtractedEntity]
    key_facts: list[str]   # ≤5 standalone factual sentences
    summary:   str         # 1–2 sentence description of the chunk
    topics:    list[str]   # ≤10 keywords / themes

class Chunk(BaseModel):
    id:              str   # UUID, used as Qdrant point ID
    text:            str   # raw chunk text
    source_doc_id:   str   # parent document UUID
    source_doc_name: str   # human-readable document name
    chunk_index:     int   # 0-based position in parent document
    extraction:      ExtractionOutput | None
```

### 3.2 Qdrant payload (per point)

Each Qdrant point stores the full chunk as a flat payload alongside the vector:

```json
{
  "text":             "The Python language was created by ...",
  "source_doc_id":    "d5e7a12b-...",
  "source_doc_name":  "python_intro",
  "chunk_index":      0,
  "summary":          "Introduction to Python's origins.",
  "topics":           ["python", "programming", "guido"],
  "key_facts":        ["Python was created by Guido van Rossum in 1991."],
  "entity_names":     ["Guido van Rossum", "Python"],
  "entity_types":     ["Person", "Product"]
}
```

Storing all retrieval-relevant data in the payload avoids a round-trip to a separate store for every search result.

### 3.3 Base entity ontology

```
Person · Organization · Location · Event · Concept · Document · Product · Date
```

These 8 types cover the vast majority of real-world knowledge. Unknown types extracted by the LLM are clamped to `Concept` in Phase 1 and promoted to first-class types in Phase 3.

### 3.4 Phase 2 graph model (Neo4j)

```cypher
// Nodes
(:Entity  {id, name, type, aliases[], canonical: true})
(:Chunk   {id, text, chunk_index})
(:Source  {id, name, authority_score})

// Edges
(:Entity)-[:MENTIONED_IN {confidence}]->(:Chunk)
(:Chunk)-[:PART_OF]->(:Source)
(:Entity)-[:RELATES_TO {
  predicate,       // "WORKS_AT", "FOUNDED", "LOCATED_IN", ...
  confidence,
  valid_from,      // ISO date or null
  valid_to,        // ISO date or null (null = currently valid)
  extracted_at,    // ingestion timestamp
  source_chunk_id
}]->(:Entity)
(:Entity)-[:CONTRADICTS {reason}]->(:Entity)  // conflict tracking
(:Entity)-[:SUPERSEDED_BY]->(:Entity)         // entity split/merge
```

---

## 4. Phase 1 — Semantic Vault

### 4.1 Architecture diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PATH                              │
│                                                                     │
│  Raw text                                                           │
│      │                                                              │
│      ▼                                                              │
│  ┌────────────┐   sentence-boundary split + overlap                 │
│  │  Chunker   │──────────────────────────────────────────────────── │
│  └────────────┘                                                     │
│      │  list[str]                                                   │
│      ▼                                                              │
│  ┌─────────────┐  Gemini tool-call → ExtractionOutput JSON          │
│  │  Extractor  │  (entities, key_facts, summary, topics)            │
│  └─────────────┘                                                    │
│      │  list[Chunk]                                                 │
│      ▼                                                              │
│  ┌─────────────┐  Gemini embed_content → 3072-dim vector            │
│  │  VectorStore│──► Qdrant upsert (vector + payload)                │
│  └─────────────┘                                                    │
│      │                                                              │
│  ┌─────────────┐  SQLite INSERT (id, name, hash, date, chunk_count) │
│  │ MetadataStore│                                                   │
│  └─────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL PATH                              │
│                                                                     │
│  User query                                                         │
│      │                                                              │
│      ▼                                                              │
│  Gemini embed_content → 3072-dim query vector                       │
│      │                                                              │
│      ▼                                                              │
│  Qdrant query_points (cosine, top-K, optional entity-type filter)   │
│      │  list[hit]  — each hit has full payload + score              │
│      ▼                                                              │
│  Gemini generate_content_stream                                     │
│    prompt = system_template + ranked_chunks + question              │
│      │  streamed tokens                                             │
│      ▼                                                              │
│  QueryResult { answer, sources[], entities_mentioned[] }            │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Chunker

**Mechanism:** Text is split on sentence-ending punctuation (`[.!?]\s+`) and paragraph breaks (`\n\n`). Sentences are accumulated into a chunk until the character budget (`CHUNK_SIZE_CHARS`, default 1 600) is exceeded, at which point the chunk is emitted and the last `CHUNK_OVERLAP_CHARS` (default 200) characters worth of sentences are carried forward.

**Why sentence-boundary splitting, not fixed-token windows?**
Fixed-token windows cut mid-sentence, which destroys the syntactic unit that gives a fact its meaning. "Guido van Rossum created — [chunk break] — Python in 1991" produces two useless half-facts. Sentence-aware splitting keeps facts intact at the cost of slightly variable chunk sizes.

**Why overlap?**
Without overlap, a fact that spans the boundary of two sentences would appear in neither chunk. Overlap ensures every sentence appears in at least one complete context window. The overlap is sentence-aligned (not character-aligned) to avoid splitting mid-sentence on the carry-forward.

**Why not semantic chunking (embedding-based)?**
Semantic chunking groups sentences by topic similarity using embeddings. It produces higher-quality chunks but requires an embedding call before extraction — doubling API cost per chunk. The sentence-boundary approach is deterministic, free, and produces chunks that are good enough for the extraction step that follows.

### 4.3 Extractor

**Mechanism:** Each chunk is sent to Gemini with `response_schema=ExtractionOutput` and `response_mime_type="application/json"`. Gemini uses its tool-calling mechanism to produce a structured JSON object that Pydantic validates. Unknown entity types are clamped to `Concept` by `_clamp_entity_types()`.

**Why structured output (tool-call) rather than free-form parsing?**
Free-form LLM output requires a regex or JSON parser that must handle malformed output. Tool-call / `response_schema` forces the model to produce structurally valid output or fail explicitly. The failure mode is an exception, not silently wrong data in the database.

**Why extraction is best-effort:**
If the LLM returns a malformed response or the API is unavailable, the chunk is still ingested — just without extraction metadata. The chunk text and its vector are always stored. Extraction metadata improves retrieval quality but is never required for the system to function.

**Retry logic:**
One retry with 2-second back-off. The Gemini free tier occasionally returns transient errors; a single retry handles >95% of these without user-visible failure.

### 4.4 Embedding

**Mechanism:** `client.models.embed_content(model="gemini-embedding-2", contents=[text], config=EmbedContentConfig(output_dimensionality=3072))` is called once per chunk during ingestion and once per query during retrieval. The 3072-dim vector is stored as a Qdrant named vector `"dense"`.

**Why Gemini embeddings instead of local models (fastembed / sentence-transformers)?**

| Dimension | Local (fastembed) | Gemini Embedding 2 |
|---|---|---|
| Dependencies | onnxruntime, numpy, tokenizers (native code) | google-genai (pure Python HTTP) |
| Python 3.14 support | Segfaults (onnxruntime ABI mismatch) | Works on any Python version |
| First-run latency | ~30s model download | Immediate |
| Embedding quality | BAAI/bge-small: moderate (384-dim) | State-of-the-art (3072-dim) |
| Cost | Free (local compute) | Free tier (generous RPM limit) |
| Offline use | Yes | No |

The crash on Python 3.14 made the choice unambiguous. Even ignoring the crash, the embedding quality of `gemini-embedding-2` at 3072 dimensions significantly exceeds `bge-small-en-v1.5` at 384 dimensions for the semantic retrieval task.

**Dimension guard:**
`VectorStore._ensure_collection()` checks that the stored Qdrant collection dimension matches `GEMINI_EMBED_DIM`. If they differ (e.g. after changing the embedding model), it raises a descriptive error rather than silently producing incorrect similarity scores.

### 4.5 Vector store (Qdrant)

**Mechanism:** Qdrant is used in local-file mode (`QdrantClient(path="./qdrant_db")`), which provides ACID-like durability without running a separate server process. For production deployment, pointing `QDRANT_URL` at `http://your-server:6333` requires no code changes.

**Search:** `query_points` with `using="dense"` performs cosine similarity. The `entity_type_filter` pushes a `FieldCondition(match=MatchValue)` filter down to the search so that Qdrant applies it before ranking (not as a post-filter), preserving the `top_k` budget.

**Why Qdrant over alternatives?**

| Criterion | Qdrant | Weaviate | Pinecone | Chroma |
|---|---|---|---|---|
| Local-file mode | ✅ | ❌ (Docker required) | ❌ (cloud only) | ✅ |
| Hybrid dense+sparse | ✅ native | ✅ | ✅ (2024+) | ❌ |
| Filter expressiveness | ✅ complex payload queries | ✅ | Limited | Limited |
| Open source | ✅ Apache 2 | ✅ BSD | ❌ | ✅ Apache 2 |
| Python 3.14 support | ✅ | Unknown | ✅ | ✅ |
| Named vectors (multi-vec) | ✅ | ✅ | Limited | ❌ |

Qdrant's local-file mode is the key differentiator for this project — it allows the full system to run on a laptop without Docker.

### 4.6 Metadata store (SQLite)

**Mechanism:** A single `source_documents` table tracks document identity (`id`, `name`, `content_hash`, `created_at`, `chunk_count`). The `content_hash` (SHA-256) enables deduplication: ingesting the same content twice raises a `ValueError` before any LLM calls are made.

**Why SQLite instead of PostgreSQL for Phase 1?**
SQLite requires zero server setup, produces a single portable file, and handles the metadata workload (small table, low concurrency, simple queries) without any operational overhead. PostgreSQL will be introduced in Phase 3 when the schema registry requires migration versioning and concurrent access from async workers.

---

## 5. Phase 2 — Knowledge Graph

### 5.1 What Phase 1 cannot do

Phase 1 answers: *"What text in my documents is most similar to this query?"*

It cannot answer:
- *"Who does Alice work with?"* — requires traversing a relationship edge
- *"Which concepts are related to Project X?"* — requires multi-hop graph traversal
- *"Are 'Alice Johnson' and 'A. Johnson' the same person?"* — requires entity resolution
- *"Document A says Alice was born in 1980; Document B says 1982 — which is right?"* — requires conflict tracking

### 5.2 Architecture additions

```
Ingestion path — new steps after extraction:

ExtractionOutput
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ENTITY RESOLUTION (3-stage)                   │
│                                                                  │
│  Stage 1 — Embedding blocking                                    │
│    For each new entity:                                          │
│      → embed entity name + context                               │
│      → ANN search in Qdrant entity index (top-5 candidates)      │
│      → cosine ≥ 0.85 → definite match (merge to canonical)       │
│      → cosine 0.65–0.85 → ambiguous (send to Stage 2)            │
│      → cosine < 0.65 → new entity (create node)                  │
│                                                                  │
│  Stage 2 — LLM disambiguation                                    │
│    For ambiguous pairs:                                          │
│      → send both entity names + surrounding context to Gemini    │
│      → binary decision: same entity? yes/no + confidence         │
│      → yes → merge to existing canonical node                    │
│      → no  → create new node                                     │
│                                                                  │
│  Stage 3 — Canonical management                                  │
│    → one UUID per resolved entity (canonical_id)                 │
│    → aliases[] list on the canonical node                        │
│    → all relationship edges use canonical_id                     │
│    → non-canonical mentions stored with ALIAS_OF edge            │
└──────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GRAPH WRITE (Neo4j)                         │
│                                                                 │
│  MERGE (:Entity {canonical_id}) SET name, type, aliases        │
│  MERGE (:Chunk {id}) SET text, chunk_index                      │
│  MERGE (:Source {id}) SET name                                  │
│  CREATE (:Entity)-[:RELATES_TO {predicate, confidence,         │
│          valid_from, valid_to, extracted_at, source_chunk_id}]  │
│       ->(:Entity)                                               │
│                                                                 │
│  If conflicting fact detected:                                  │
│    CREATE (:Entity)-[:CONTRADICTS {reason}]->(:Entity)          │
└─────────────────────────────────────────────────────────────────┘

Retrieval path — new branch:

User query
      │
      ▼
Query classifier (Gemini): relationship query? factual query?
      │
      ├─── factual → vector search (Phase 1 path)
      │
      └─── relationship → graph traversal
               │
               ▼
           Cypher query generation (Gemini → Cypher string)
               │
               ▼
           Neo4j execution (1–3 hops, filtered by confidence)
               │
               ▼
           Graph results + vector results → merge → synthesise
```

### 5.3 Entity resolution rationale

**Why three stages instead of one?**

A single embedding similarity check misses cases where two entities have very similar names but are different (e.g. two politicians named "John Smith"). A single LLM call for every pair is too expensive at scale — an O(n²) cost for n entities.

The 3-stage funnel:
1. Embedding blocking reduces the candidate pairs from O(n²) to O(n log n)
2. LLM disambiguation handles only the ambiguous band (0.65–0.85 cosine)
3. Canonical management ensures all future references resolve correctly

This mirrors the entity resolution literature (blocking → classification → canonicalisation).

**Why 0.85 / 0.65 thresholds?**
These are empirically calibrated for person and organisation names in English. Names with cosine ≥ 0.85 (e.g. "Guido van Rossum" vs "Guido Van Rossum") are almost always the same entity. Names with cosine < 0.65 (e.g. "Guido van Rossum" vs "Python Software Foundation") are almost always different. The 0.65–0.85 band is genuinely ambiguous and warrants an LLM call.

### 5.4 Bi-temporal fact tracking

Every relationship edge stores two time axes:
- **Event time** (`valid_from`, `valid_to`): when the fact was true in the world
- **Ingestion time** (`extracted_at`): when the system learned it

This allows queries like "what did we know about Alice as of January 2024?" without conflating knowledge-time with fact-time. The pattern is borrowed from temporal databases (Snodgrass 1999) and from Zep's knowledge graph design.

**Conflict handling:**
When a new extraction contradicts an existing edge (same subject, predicate, and object but different value — e.g. different birthyear):
1. Both facts are stored. The existing edge is not modified.
2. A `CONTRADICTS` edge links the two conflicting fact nodes.
3. At query time, the synthesis prompt receives both versions with their provenance (source, date, confidence) and is instructed to report the conflict if unresolvable.

**Why not last-write-wins?**
LWW silently discards prior information. If Document A says Alice was born in 1980 and Document B (lower authority, later ingested) says 1982, LWW would lose the better answer. Provenance-preserving storage lets the system — and the user — reason about which source to trust.

### 5.5 Why Neo4j

| Criterion | Neo4j | Amazon Neptune | ArangoDB | DuckDB + edge table |
|---|---|---|---|---|
| LLM tooling | ✅ LangChain, LlamaIndex, GraphRAG native | Limited | Limited | None |
| Cypher language | ✅ widely known | ✅ (Gremlin + openCypher) | AQL (proprietary) | SQL |
| Local mode | ✅ embedded / Community | ❌ cloud only | ✅ | ✅ |
| Free tier | ✅ Community Edition | ❌ | ✅ | ✅ |
| Graph algorithms | ✅ GDS library | Limited | ✅ | None |
| Python driver | ✅ neo4j-python-driver | ✅ | ✅ | ✅ |

Neo4j's Cypher language and its direct integration with the LLM ecosystem (Text2Cypher, GraphRAG) make it the clear choice. The Community Edition is free and runs locally without a licence.

---

## 6. Phase 3 — Adaptive System

### 6.1 What Phase 2 cannot do

Phase 2 answers questions about relationships between known entity types.

It cannot:
- Recognise that the corpus has evolved to contain a new domain (e.g. medical records after the corpus previously contained only business documents) and introduce appropriate new types (`Drug`, `Symptom`, `Treatment`)
- Handle queries that span multiple retrieval strategies without manual routing
- Score sources by authority (a peer-reviewed paper vs a tweet)
- Retroactively re-extract documents when the schema evolves

### 6.2 Ontology evolution engine

```
Background worker (runs every N new documents or on schedule):

1. Collect all entities with type "Concept" (the catch-all for unknown types)
2. Cluster their embeddings using Leiden algorithm (community detection)
3. For each cluster with ≥ MIN_CLUSTER_SIZE members:
     → Sample cluster members → send to Gemini
     → Gemini proposes a canonical type name + description
     → Check against existing type hierarchy:
         - Is this a subtype of an existing type? → add as child
         - Is this genuinely new? → add at root with budget check
4. If budget allows (MAX_TYPES_PER_LEVEL not exceeded):
     → Add to schema registry (PostgreSQL) with schema_version + 1
     → Queue retroactive re-extraction jobs for affected documents
5. Affected documents: those whose entity_names intersect the cluster members
   (targeted, not full-corpus)
```

**Ontology budget / explosion prevention:**

An unconstrained system would eventually create thousands of types (e.g. one per scientific subspecialty). This degrades query quality and increases extraction cost. The budget enforces:
- `MAX_TYPES_PER_LEVEL = 20` — at most 20 sibling types at any depth
- New types must fit under an existing parent OR displace a low-usage type
- Type subsumption hierarchy is mandatory: `Drug → Compound → Substance`, not three flat siblings

**Why cluster-then-propose instead of LLM-proposes-on-each-chunk?**
Per-chunk type proposals would produce inconsistent names ("pharmaceutical agent", "medication", "drug", "medicine" as four separate types for the same concept). Clustering first groups co-occurring concepts, then asks the LLM to name the cluster — producing stable, canonical type names.

### 6.3 Agentic retrieval

```
User query
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  PLANNER AGENT                                                   │
│    → Decompose query into sub-questions                          │
│    → Classify each sub-question:                                 │
│        semantic   → dense vector search                          │
│        keyword    → BM25 sparse search (Phase 3 adds this back) │
│        relational → Neo4j Cypher traversal                       │
│        structured → PostgreSQL (date ranges, exact field match)  │
│    → Estimate hop depth for relational questions (1, 2, or 3)    │
└──────────────────────────────────────────────────────────────────┘
      │  retrieval plan: list[(strategy, sub-question)]
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  RETRIEVER AGENTS (parallel)                                     │
│    Dense:      Qdrant cosine top-K                               │
│    Graph:      Gemini → Cypher → Neo4j → entity + chunk results  │
│    Structured: PostgreSQL (e.g. "events after 2023-01-01")       │
└──────────────────────────────────────────────────────────────────┘
      │  raw results from each path
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  FUSION + RERANKING                                              │
│    RRF (Reciprocal Rank Fusion) across all result lists          │
│    ColBERT or Gemini-as-judge reranker on top-20                 │
│    Conflict detection: flag chunks with CONTRADICTS edges        │
└──────────────────────────────────────────────────────────────────┘
      │  top-K fused, ranked, conflict-annotated chunks
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  SYNTHESISER AGENT                                               │
│    → Generate answer with inline citations [Source: name, chunk] │
│    → If conflict detected: present both versions with provenance │
│    → Source authority scores weight the synthesis prompt         │
└──────────────────────────────────────────────────────────────────┘
```

**Why Reciprocal Rank Fusion for multi-path fusion?**
RRF is parameter-free, rank-based (immune to score scale differences between paths), and empirically robust. The formula `1 / (k + rank)` (typically k=60) gives diminishing returns to top-ranked results, which prevents one retrieval path from dominating even when its raw scores are much larger than another's.

Learned fusion (CrossEncoder, ColBERT) outperforms RRF but requires a training dataset and adds significant latency. RRF is used first; ColBERT reranking is applied as a final pass on the RRF output (top-20 → top-8).

### 6.4 Source authority scoring

Each source document accumulates a trust score:

```
authority_score = (
  corroboration_count    # how many other sources confirm its facts
  / (corroboration_count + contradiction_count + 1)
) × recency_weight      # newer sources decay slower
```

Authority scores influence synthesis by:
1. Downweighting chunks from low-authority sources in the synthesis prompt ordering
2. Being reported inline: "According to [high-authority source]..."
3. Breaking ties in conflict resolution

**Why not a fixed trust hierarchy (e.g. papers > blogs > tweets)?**
A fixed hierarchy requires manually tagging each source with a tier, which defeats the purpose of an automated system. The corroboration-based score is self-organising: sources that agree with the majority of other sources earn authority; isolated claims score low.

### 6.5 Infrastructure additions in Phase 3

**PostgreSQL replaces SQLite** for the metadata/schema registry because:
- Schema versioning with migrations requires atomic DDL (SQLite has limited ALTER TABLE)
- Async workers need concurrent writes (SQLite has file-level write lock)
- The schema registry table has foreign key relationships that benefit from RDBMS constraints

**Redis Streams / Celery** for the job queue because:
- Retroactive re-extraction jobs are long-running (minutes per document batch)
- They must not block the main ingestion path
- Jobs need retry semantics and progress tracking

---

## 7. Technology Decisions

### 7.1 LLM provider — Gemini vs Anthropic vs OpenAI

| Criterion | Gemini 3 Flash | Claude Sonnet | GPT-4o |
|---|---|---|---|
| Free tier | ✅ generous | ❌ API-only | ❌ API-only |
| Structured output | ✅ response_schema | ✅ tool_use | ✅ tool_use |
| Streaming | ✅ generate_content_stream | ✅ | ✅ |
| Embedding model | ✅ gemini-embedding-2 | ❌ (use Voyage AI) | ✅ text-embedding-3 |
| Context window | 1M tokens | 200K tokens | 128K tokens |
| Python SDK (new) | google-genai | anthropic | openai |

**Decision: Gemini.** The free tier eliminates cost friction during development and for personal use. Having extraction, synthesis, and embeddings all in one provider simplifies auth and billing. The 1M context window means full documents rarely need chunking for extraction.

### 7.2 Embedding model

| Model | Dims | Quality | Cost | Native deps |
|---|---|---|---|---|
| `gemini-embedding-2` | 3072 | State-of-the-art | Free tier | None |
| `gemini-embedding-001` | 3072 | Excellent | Free tier | None |
| `text-embedding-004` | 768 | Very good | Free tier | None |
| `BAAI/bge-small-en-v1.5` | 384 | Good | Free (local) | onnxruntime (crashes on Py 3.14) |
| `text-embedding-3-large` | 3072 | Excellent | Paid | None |
| `voyage-3` | 1024 | Excellent | Paid | None |

**Decision: `gemini-embedding-2`.** Highest quality available on the free tier with zero native dependencies.

### 7.3 Retrieval strategy — dense vs hybrid

| Approach | Dense only | Dense + BM25 (hybrid) | Dense + BM25 + Graph |
|---|---|---|---|
| Semantic queries | Excellent | Excellent | Excellent |
| Keyword/exact queries | Good | Excellent | Excellent |
| Relationship queries | Poor | Poor | Excellent |
| Implementation complexity | Low | Medium | High |
| Phase | 1 | 3 | 2–3 |

Phase 1 uses dense-only because the embedding quality of `gemini-embedding-2` at 3072 dims handles the vast majority of semantic queries without needing BM25. Hybrid search is re-introduced in Phase 3 using Qdrant's native sparse vector support (previously blocked by fastembed/onnxruntime crash).

### 7.4 Web UI — Gradio vs Streamlit vs FastAPI+HTML

| Criterion | Gradio 6 | Streamlit | FastAPI + HTML |
|---|---|---|---|
| Lines of code | ~150 | ~200 | ~500+ |
| Streaming chat component | ✅ native | ⚠️ manual | Manual |
| LLM/AI idioms (chat, file upload) | ✅ built-in | ✅ built-in | Manual |
| Layout control | Good | Good | Complete |
| Dependency weight | Medium | Medium | Light |
| Custom CSS | Limited | Limited | Complete |

**Decision: Gradio.** The streaming chat component and two-column `gr.Blocks` layout map directly to the use case. Gradio 6's new Blocks API provides enough layout control without requiring frontend code.

### 7.5 Chunking strategy — sentence vs token vs semantic

| Strategy | Quality | Cost | Determinism |
|---|---|---|---|
| Fixed-token windows | Low (cuts mid-sentence) | Free | Yes |
| Sentence-boundary with overlap | Good | Free | Yes |
| Semantic chunking (embedding-based) | Excellent | 1 embed/sentence | Yes |
| LLM-based segmentation | Excellent | 1 LLM call/doc | Yes |

**Decision: Sentence-boundary with overlap.** Free, deterministic, and produces complete factual units. Semantic chunking is a future upgrade that would improve retrieval quality at the cost of doubling embedding API calls during ingestion.

---

## 8. Cross-Cutting Concerns

### 8.1 Deduplication

Content deduplication uses SHA-256 of the raw text. This prevents:
- Double-ingestion of the same document
- Wasted LLM calls on already-processed content
- Doubled chunks in retrieval results

The hash is checked against `source_documents.content_hash` (UNIQUE constraint) before any LLM calls. Failed deduplication raises `ValueError` immediately.

### 8.2 Error handling philosophy

```
Extraction failure   → log warning, store chunk without metadata (best-effort)
Embedding failure    → retry ×3 with 1.5× back-off, then raise
Qdrant write failure → propagate exception (data would be lost silently otherwise)
SQLite write failure → propagate exception
Graph write failure  → propagate exception (Phase 2+)
Synthesis failure    → propagate exception (user sees error, no hallucinated answer)
```

Extraction and embedding fail differently because extraction is enhancement (missing it degrades quality) while embedding is load-bearing (missing it means the chunk is unretrievable).

### 8.3 Testing architecture

```
Unit tests (no I/O):
  test_chunker.py       — pure function, no fixtures
  test_extractor.py     — Gemini client replaced with MagicMock

Integration tests (in-process I/O only):
  test_storage.py       — real Qdrant :memory:, embed_fn=mock_embed (no API)
  test_ingestion.py     — real Qdrant :memory:, Gemini mocked
  test_retrieval.py     — real Qdrant :memory:, Gemini mocked

End-to-end tests (real API):
  test_e2e.py           — full pipeline, skipped if GEMINI_API_KEY not set
```

The `embed_fn` parameter on `VectorStore` is the key injection point: it replaces the Gemini embedding API with a deterministic pure-Python function during tests. No monkey-patching, no `importlib` tricks.

Mock embedding function (`conftest._mock_embed`) is:
- **Deterministic**: same text always produces the same vector (hash-seeded RNG)
- **Normalised**: unit-length vectors, valid for cosine similarity
- **Dimensionally correct**: 3072-dim, matching production

### 8.4 Configuration management

All runtime parameters are in `semantic_vault/config.py` via `pydantic-settings`. The settings object reads from:
1. Environment variables (highest priority)
2. `.env` file
3. Default values (lowest priority)

This means CI/CD can override settings without modifying files, `.env` handles local development, and defaults make the system work out of the box.

---

## 9. Performance Characteristics

### 9.1 Ingestion latency (per document)

| Step | Typical latency | Bottleneck |
|---|---|---|
| Chunking | < 10ms | CPU |
| Extraction (per chunk) | 400–800ms | Gemini API round-trip |
| Embedding (per chunk) | 200–400ms | Gemini API round-trip |
| Qdrant upsert | 5–20ms | Local disk |
| SQLite write | < 5ms | Local disk |
| **Total (1 chunk)** | **~1s** | Gemini API |
| **Total (10 chunks)** | **~10s** | Gemini API (sequential) |

Extraction and embedding are currently called sequentially per chunk. Phase 3 will introduce async concurrency (asyncio + `google-genai` async client) to parallelize across chunks, reducing 10-chunk ingestion from ~10s to ~2s.

### 9.2 Retrieval latency

| Step | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Query embedding | ~300ms | ~300ms | ~300ms |
| Vector search (Qdrant) | < 50ms | < 50ms | < 50ms |
| Graph traversal (Neo4j) | — | 50–200ms | 50–200ms |
| RRF fusion | — | — | < 5ms |
| Reranking | — | — | 100–300ms |
| Synthesis (streaming) | 500ms–2s | 500ms–2s | 500ms–2s |
| **Total (p50)** | **~1s** | **~1.5s** | **~2s** |

Synthesis latency is the dominant term and is bounded by the Gemini API, not local infrastructure.

### 9.3 Scalability

| Metric | Phase 1 ceiling | Mitigation |
|---|---|---|
| Documents | ~10K (SQLite OK) | Phase 3: PostgreSQL |
| Chunks | ~100K (Qdrant local OK) | Server mode: Qdrant Cloud or self-hosted |
| Concurrent users | 1 (Gradio default, single process) | Multiple workers: Gunicorn + Gradio queue |
| Gemini free tier RPM | 15 RPM (Gemini 3 Flash) | Add retry/queue; upgrade tier |
| Entity count (Neo4j) | ~1M nodes (Community OK) | Read replicas; index on canonical_id |
