# CLAUDE.md — Semantic Vault

A three-phase LLM-powered knowledge management system. All phases are production-complete. This file is the authoritative technical reference for contributors and AI assistants.

---

## Running the project

```bash
source .venv/bin/activate          # always activate the project venv first

python app.py                      # Gradio UI at http://127.0.0.1:7860
python main.py ingest "text"       # add content
python main.py query "question"    # ask a question
python main.py list                # list all documents
python main.py rebuild-graph       # re-index existing docs into Neo4j

pytest tests/ --ignore=tests/test_e2e.py -v   # 138 unit+integration tests
pytest tests/test_e2e.py -v                   # live API tests (needs GEMINI_API_KEY)

pip install -e ".[ui,dev]"         # install all dependencies
```

---

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | All runtime settings via pydantic-settings (reads `.env`) |
| `models.py` | `ExtractedEntity`, `ExtractedRelationship`, `ExtractionOutput`, `Chunk`, `SourceDocument`, `QueryResult` |
| `chunker.py` | `chunk_text()` — sentence-boundary splitting, overlap, no external deps |
| `extractor.py` | `extract()` — Gemini structured JSON output (`response_schema=ExtractionOutput`); normalises entity types and relationship predicates |
| `storage.py` | `VectorStore` (Qdrant + Gemini embeddings) · `EntityIndex` (ANN blocking) · `MetadataStore` (SQLite) |
| `ingestion.py` | `ingest_text()` — orchestrates chunk → extract → embed → store (Phase 1) + entity resolve → graph write (Phase 2) |
| `retrieval.py` | `retrieve_and_synthesize()` — vector search (Phase 1) + graph traversal (Phase 2) + parallel retrieval + RRF + reranking + authority (Phase 3) |
| `entity_resolver.py` | `EntityResolver` — 3-stage: embedding blocking → Gemini LLM disambiguation → canonical UUID management |
| `graph_store.py` | `GraphStore` — Neo4j CRUD: `upsert_entity`, `add_alias`, `upsert_relationship`, `traverse`, `text_to_cypher_query` |
| `query_planner.py` | `QueryPlanner` — Gemini decomposes questions, classifies intent, selects retrieval strategies |
| `reranker.py` | `rrf_fuse()` (Reciprocal Rank Fusion) · `llm_rerank()` (Gemini scores chunks 1–5) |
| `authority_scorer.py` | `AuthorityScorer` — per-doc trust = corroborations / (corroborations + contradictions + 1) |
| `ontology_engine.py` | `OntologyEngine` — cosine-threshold clustering of Concept entities → Gemini proposes type names → `SchemaRegistry` |
| `app.py` | Gradio 6 two-panel web UI; streaming synthesis; Phase 2+3 features auto-activated from settings |
| `main.py` | argparse CLI: ingest · query · list · rebuild-graph |

---

## Data flow

### Ingestion
```
ingest_text()
  → chunk_text()                         sentence-boundary split + overlap
  → extract()                            Gemini: entities, relationships, facts, summary, topics
  → [Phase 2] EntityResolver.resolve()   3-stage entity resolution per entity
  → [Phase 2] GraphStore.upsert_entity() create/merge Neo4j node
  → [Phase 2] GraphStore.upsert_relationship()  write typed edge with provenance
  → VectorStore.upsert_chunks()          Gemini embed → Qdrant (payload includes full extraction)
  → MetadataStore.save_document()        SQLite row
  → [Phase 3] AuthorityScorer.record_ingestion()
  → [Phase 3 background] OntologyEngine.run_evolution()  every N documents
```

### Retrieval
```
retrieve_and_synthesize()
  → [Phase 3] QueryPlanner.plan()        decompose + classify + select strategies
  → Parallel ThreadPoolExecutor:
      dense:      VectorStore.hybrid_search()  per sub-question
      graph:      GraphStore.text_to_cypher_query()  if relational
      structured: (future) PostgreSQL filter
  → rrf_fuse()                           merge ranked lists (Phase 3) or direct slice (Phase 1/2)
  → [Phase 3] llm_rerank()              score top-20 → return top-K
  → _SYNTHESIS_PROMPT.format()          chunks + graph section + authority section
  → Gemini generate_content()           final answer with inline citations
```

---

## Testing conventions

### No real API calls in unit/integration tests
- `vector_store` fixture: in-memory Qdrant + `embed_fn=_mock_embed` (deterministic 3072-dim unit vectors)
- `entity_index` fixture: real `EntityIndex` using the mock `vector_store`
- `mock_graph_store` fixture: `MagicMock()` — no Neo4j required
- Gemini extraction: `MagicMock` returning a pre-built `ExtractionOutput`
- Gemini synthesis: `MagicMock` returning a fixed string

### Key injection points (dependency injection pattern)
```python
VectorStore(embed_fn=_mock_embed)          # bypasses Gemini embedding API
EntityResolver(_llm_fn=lambda a,b,c: False)  # bypasses Gemini disambiguation
GraphStore(_driver=mock_driver)            # bypasses Neo4j
QueryPlanner(gemini_client=mock_client)   # bypasses Gemini query planning
```

### End-to-end tests (`test_e2e.py`)
All decorated with `@_NEEDS_KEY` — skipped automatically when `GEMINI_API_KEY` is unset. Run with a real key to validate the live pipeline.

### The mock embedding function
`_mock_embed(texts)` in `conftest.py` uses `random.Random(hash(text) & 0xFFFF)` — same text always produces the same unit vector. **The same entity name always produces the same vector** (context is not included), which is essential for entity resolution tests.

---

## Important constraints

### 1. Embedding dimension must match Qdrant
`VectorStore._ensure_collection()` checks that the stored dimension equals `GEMINI_EMBED_DIM`. If you change the model, delete `qdrant_db/` to reset. The error message says exactly this.

### 2. Never import fastembed or onnxruntime
They segfault on Python 3.14. Embeddings go through the Gemini API only.

### 3. Extraction is best-effort
`ingest_text()` catches extraction exceptions and stores chunks without metadata. Do not make extraction mandatory — the vector is always stored even when extraction fails.

### 4. Graph writes are best-effort
`_process_graph()` wraps every write in try/except. Graph failures print a warning; ingestion always completes.

### 5. Phase 3 is opt-in
All Phase 3 features are activated by optional parameters or env flags. Calling `retrieve_and_synthesize(query, vector_store)` with no extra args is identical to Phase 1.

### 6. Reranker needs more candidates than top_k
When `ENABLE_RERANKING=true`, retrieval fetches `max(RERANK_TOP_N, k*2)` candidates before reranking down to `top_k`. This is enforced in `retrieval.py` — do not change the search call to use `k` directly when reranking.

### 7. Cypher uses ON CREATE / ON MATCH (not CALL subquery)
`graph_store.upsert_entity()` uses `ON CREATE SET ... ON MATCH SET ... [a IN aliases WHERE NOT a IN e.aliases]`. Do not revert to the `CALL {}` subquery syntax — it generates deprecation warnings in Neo4j 2026.04.

### 8. Entity embedding uses name + type only (no context)
`EntityIndex.upsert()` embeds `"{name} ({type})"` without the surrounding chunk text. This is intentional: the same entity must always map to the same vector regardless of which document it appears in. Context is only used in the Gemini LLM disambiguation stage.

---

## Phase 3 activation in app.py

The web UI reads settings at startup. To activate Phase 3 features, add to `.env`:

```bash
ENABLE_QUERY_PLANNER=true
ENABLE_RERANKING=true
ENABLE_AUTHORITY=true
```

Then restart `python app.py`. The header shows active features.

## Ontology evolution

Run manually via CLI or triggered automatically (every `ONTOLOGY_EVOLUTION_INTERVAL` ingestions):

```bash
python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from semantic_vault.storage import VectorStore, MetadataStore
from semantic_vault.storage import EntityIndex
from semantic_vault.ontology_engine import OntologyEngine, SchemaRegistry
from semantic_vault.config import settings

vs = VectorStore(url=settings.qdrant_url)
ei = EntityIndex(vs)
sr = SchemaRegistry(db_path=settings.db_path)
engine = OntologyEngine(ei, sr)
new_types = engine.run_evolution()
print('New types:', new_types)
print('Registry:', [t['type_name'] for t in sr.list_types()])
"
```
