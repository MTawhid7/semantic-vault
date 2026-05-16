# Limitations Report — Semantic Vault

> A critical assessment of known and anticipated limitations against the project's
> core promise: *users add unstructured data without schema design; the system
> automatically understands, structures, relates, and retrieves it intelligently.*

---

## 1. Performance

### 1.1 Ingestion is slow and rate-limited

**Observed:** A single document with ~10 entities takes 1–3 minutes to ingest.

**Root cause breakdown:**

| Step | API calls | Estimated time (free tier) |
|---|---|---|
| Gemini extraction (1 call per chunk) | 1 | ~1–2s |
| Entity embedding — search (1 per entity) | 10 | ~4s |
| Entity embedding — upsert (1 per entity, partially cached) | 0–10 | 0–4s |
| Chunk embedding | 1 | ~400ms |
| Neo4j AuraDB writes (entity nodes + relationship edges) | 15–25 | ~2–4s |
| **Gemini free-tier rate limit backoff (15 RPM)** | — | **dominant: 30–90s** |

The rate limit is the actual bottleneck. Gemini 3 Flash Preview allows 15 requests per minute. A document that triggers 13 Gemini calls cannot complete in under 52 seconds even with zero processing time. Documents with many entities routinely exceed 2 minutes.

**No async ingestion pipeline exists.** The user cannot add a second document while the first is being processed. The UI blocks.

### 1.2 Retrieval latency is high with Phase 3 active

**Observed:** 30–45 seconds for complex relational queries.

**Breakdown:**
- Query planner call (Gemini): ~1–2s
- Parallel dense retrieval (N sub-questions): ~1s (fast)
- Graph Text2Cypher: Gemini call (~1–2s) + AuraDB execution (~100–500ms cloud latency)
- LLM reranking (if enabled): ~2–5s for Gemini to score 20 chunks
- Synthesis streaming start: ~2–3s before first token

Total: 7–14s before first token. With rate-limit backoff, this can reach 30–45s. The streaming UI gives the impression of waiting even though tokens trickle through.

### 1.3 AuraDB free tier auto-pauses after 72 hours of inactivity

The first query after a pause takes 30–60 seconds to wake the instance. Users will perceive this as a crash. There is no keepalive mechanism and no graceful handling of the paused state in the application.

### 1.4 The embedding cache is in-process and lost on restart

The `_embed_cache` dict lives in the `VectorStore` object. Every restart re-embeds everything from scratch. For entity resolution, the same entities are re-embedded on every `python app.py` invocation.

---

## 2. Knowledge Quality and Accuracy

### 2.1 Context pollution across ingestion sessions

**Observed:** When documents from two different corpora share entity names (e.g., "Elena Vasquez" as MIT professor in Corpus A and as Helix CEO in Corpus B), the retrieval system mixes them into the same answer, producing hallucinated conflicts.

**Root cause:** There is no corpus isolation. All documents from all sessions share a single global Qdrant collection, SQLite database, and Neo4j graph. There is no `workspace_id`, `session_id`, or `user_id` filtering at any layer. Adding a document today contaminates queries asked six months from now if the vocabulary overlaps.

**Severity:** High. This is a fundamental design gap that directly violates the "just add data" promise — users must manage their own data hygiene by clearing the entire store before starting a new context.

### 2.2 Relationship extraction is unreliable

Gemini extracts relationships from individual chunks, not the full document. A relationship whose subject and object appear in different chunks (e.g., "Alice joined the company in 2020" and "Alice is the head of the engineering team" are two separate chunks) may never be extracted as a single edge. The `_normalise()` function discards any relationship whose endpoints are not both in the current chunk's entity list.

Relationship predicates are free-form strings normalised to `ALL_CAPS_UNDERSCORED`. Gemini invents inconsistent predicates: `WORKS_AT`, `IS_EMPLOYED_BY`, `EMPLOYED_AT`, and `JOINED` might all express the same relationship across different chunks, producing duplicate edges in the graph.

### 2.3 Entity resolution makes permanent, uncorrectable decisions

When `EntityResolver` decides that "Alice" and "Alice Johnson" are the same entity, they are merged under one canonical UUID. This decision cannot be reversed without rebuilding the entire graph. There is no correction interface, no confidence threshold that triggers human review, and no audit log of merge decisions.

The blocking threshold (0.85 cosine) was designed for English proper nouns. It performs poorly on:
- Acronyms ("MIT" vs "M.I.T.")
- Non-English names with alternative transliterations
- Short names with high embedding similarity to unrelated entities ("Kim" the person vs "Kim" the country abbreviation)

### 2.4 The LLM synthesis over-reports conflicts

**Observed:** Answers about clinical drug trials include unprompted paragraphs about founding year discrepancies. Answers about multi-hop connections append summaries of every known inconsistency in the corpus.

The synthesis prompt instructs the model to "flag conflicts." The model interprets this as "flag all conflicts in all retrieved chunks regardless of relevance to the question." This makes answers verbose, harder to read, and less useful.

### 2.5 Hallucination in extraction is not detected

When Gemini extracts an entity or relationship that does not exist in the source text, there is no verification step. The only guard is `_clamp_entity_types()` (normalises unknown types to "Concept") and `_normalise()` (discards relationships whose endpoints aren't in the entities list). A fabricated entity name passes both checks and enters the knowledge graph as ground truth.

### 2.6 Chunking can split atomic facts

The sentence-boundary chunker produces ~1600-character chunks with 200-character overlap. A fact that spans two sentences near a chunk boundary (e.g., a long quote that begins in one chunk and ends in another) may be split, with neither half being meaningful in isolation. The relationship extractor will miss the connection because both entities must appear in the same chunk.

---

## 3. Retrieval Accuracy

### 3.1 Inventory queries are fundamentally mismatched to the retrieval model

**Observed:** "Which organisations are mentioned?" returns 7 out of 13 expected organisations with `top_k=8`.

Vector search returns the K most *similar* chunks to the query, not the K most *complete*. A query asking for an exhaustive list of all X needs full corpus coverage, which cosine similarity cannot provide. No retrieval K is sufficient for arbitrary inventories over large corpora.

### 3.2 The relational query classifier is a heuristic, not a model

`_is_relational_query()` is a compiled regex over ~15 keywords. It misses:
- "What organizations is Alice affiliated with?" (no keyword match)
- "Give me the team around Dr. Chen" (no keyword match)
- "Trace the connection from the Warriors to Helix" (no keyword match)

It false-positives on:
- "What did the company report to shareholders?" (matches "report")
- "What company belongs in this sector?" (matches "belongs")

When the classifier fires incorrectly, a slow Text2Cypher call is made for a query that would have been answered better by pure vector search.

### 3.3 The knowledge graph has a cold-start problem

Entities and relationships are written to Neo4j only at ingestion time. If Neo4j is not connected when documents are added, the graph is empty. The `rebuild-graph` command exists as a workaround, but it requires re-extraction (additional Gemini calls) for documents ingested without the graph connection.

There is no visual indication in the UI that the graph is empty, so users may ask relational questions and receive vector-only answers without realising the graph is not being consulted.

### 3.4 Text2Cypher has no safety guarantees

The Cypher generated by Gemini is constrained only by a prompt instruction ("use only MATCH and RETURN"). There is no parameterisation, no query plan validation, and no read-only enforcement at the driver level. A sufficiently creative prompt injection in ingested content could influence generated Cypher.

When Cypher fails (syntax error, schema mismatch), `text_to_cypher_query()` returns an empty list silently. The user receives a vector-only answer with no indication that the graph retrieval failed.

### 3.5 Answers have no conversation memory

Each question is processed independently. The system cannot answer:
- "Tell me more about that."
- "What about the second company you mentioned?"
- "Compare the last two answers."

Every query starts from scratch against the knowledge base. The chat history displayed in the UI is cosmetic — it is not passed to the retrieval pipeline.

---

## 4. Ontology and Schema

### 4.1 The base ontology is fixed at 8 types and is not consulted during extraction

Gemini is prompted to use `Person, Organization, Location, Event, Concept, Document, Product, Date`. The `SchemaRegistry` SQLite table tracks additional types proposed by the evolution engine, but the extraction prompt is never updated to include them. New types exist in the registry but are never used in future extractions — the system self-contradicts.

### 4.2 Ontology evolution has no retroactive effect

When the evolution engine proposes a new type (e.g., `SportsTeam`), existing entities that should be `SportsTeam` remain tagged as `Concept` in Qdrant and Neo4j. A "retroactive re-extraction" job is described in the architecture but not implemented. The registry grows; the data does not change.

### 4.3 Cluster quality depends on corpus size and distribution

The union-find clustering requires `MIN_CLUSTER_SIZE = 4` members before proposing a new type. In corpora with fewer than 20 documents, most specialty entity clusters have 1–3 members and are never promoted. The ontology only evolves meaningfully at scale.

Cosine threshold clustering (0.75) is sensitive to the embedding model's representation. Entities that are semantically related but differently expressed ("aspirin" vs "acetylsalicylic acid") may not cluster together if the embedding model places them far apart.

---

## 5. Data Management

### 5.1 Documents cannot be deleted or updated

There is no `delete`, `update`, or `replace` operation for any layer (Qdrant, SQLite, Neo4j). To correct a mistake, the user must clear the entire database and re-ingest. This is incompatible with the "continuously add" promise — adding a correction requires destroying everything.

### 5.2 No support for non-text content

The system accepts only plaintext. PDF, Word, image (OCR), audio (transcription), HTML, CSV, and JSON files must be pre-converted by the user before ingestion. The "add anything" promise is limited to text.

### 5.3 The SQLite database and Qdrant store are not transactionally linked

If Qdrant upsert succeeds but SQLite `save_document` fails (e.g., disk full), the document's chunks exist in the vector store but the document record does not exist in the metadata store. The deduplication check (`document_exists(content_hash)`) will not prevent re-ingestion on the next attempt, resulting in duplicate chunks.

### 5.4 No user or access management

The Gradio UI is unauthenticated. Any user on the network can access it, add documents, and query the knowledge base. All users share a single global knowledge base with no partitioning. This makes the system unusable in any multi-tenant or shared-access scenario.

---

## 6. Scalability

| Limit | Tier | Ceiling |
|---|---|---|
| Gemini generation model | Free | 15 requests/minute |
| Gemini embedding model | Free | ~1500 requests/day (varies) |
| AuraDB Free nodes | Free | 200,000 |
| AuraDB Free relationships | Free | 400,000 |
| Qdrant local file | Single-process | ~10M vectors before disk I/O dominates |
| SQLite | Single-writer | ~10K documents before lock contention |
| Gradio | Single-process | 1 concurrent user (default) |
| Embedding cache | In-process dict | Unbounded memory growth |

At 10 entities + 5 relationships per chunk, 400K relationships ≈ 80,000 chunks ≈ 8,000 documents before AuraDB Free is saturated. A medium corporate knowledge base with 100,000 documents is an order of magnitude beyond the free tier ceiling.

---

## 7. Specific to the "No Schema Design" Promise

The following are cases where the system silently requires user knowledge that contradicts the "automatic understanding" promise:

| Expectation | Reality |
|---|---|
| Add documents from any domain and they integrate | Multiple domains share one namespace — entity name collisions produce hallucinated conflicts |
| The system resolves the same entity under different names | Only within the 0.65–0.85 cosine band AND if the LLM disambiguation fires. Short or ambiguous names fail. |
| The system detects conflicting facts | Only if both conflicting facts appear in retrieved chunks for a given query. Conflicts that never co-occur in top-K results are invisible. |
| Knowledge grows smarter as more is added | True for vector search. The graph requires explicit ingestion-time connectivity. The ontology requires manual evolution trigger. |
| Users never need to think about structure | Users must clear the store between unrelated corpora, run `rebuild-graph` when Neo4j is added, and run ontology evolution manually. |

---

## 8. Summary Priority Matrix

| Limitation | Impact | Fix complexity |
|---|---|---|
| Context pollution (no corpus isolation) | Critical | Medium — add `corpus_id` filter to all queries |
| No document deletion | High | Medium — Qdrant + Neo4j + SQLite delete APIs |
| Gemini rate limiting blocks ingestion | High | Low–Medium — async queue with back-off |
| Cold-start graph (empty on first use) | High | Low — UI warning; auto-trigger rebuild on Neo4j connect |
| Conflict over-reporting in synthesis | Medium | Low — add "only flag relevant conflicts" to prompt |
| Inventory queries miss items | Medium | Medium — graph entity roster endpoint for aggregation |
| No conversation memory | Medium | Medium — pass last N turns as context |
| Non-text content unsupported | Medium | Medium — document conversion layer |
| Entity resolution irreversible | Medium | High — correction interface + re-resolution |
| Ontology schema not used in extraction | Medium | Medium — rebuild prompt on each evolution cycle |
| No document update | High | Medium — delete + re-ingest with same name |
| No authentication | High | Low — Gradio `auth=` parameter |
| AuraDB auto-pause | Low | Low — keepalive ping + reconnect logic |
| Embedding cache lost on restart | Low | Low — persist cache to SQLite |
