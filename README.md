# Semantic Vault

An intelligent knowledge base where you add raw text and ask questions — no schema design, no SQL, no manual tagging. Three LLM-powered phases handle structure, relationships, and retrieval automatically.

---

## What it does

| You do | The system does |
|---|---|
| Paste notes, articles, research | Chunks, extracts entities + relationships, embeds |
| Ask a question | Plans retrieval strategy, searches vectors + graph, reranks, synthesises |
| Add more documents | Resolves entities to existing canonical nodes, detects conflicts, evolves the ontology |

---

## Quick start

### 1. Prerequisites

- Python 3.11+
- [Gemini API key](https://aistudio.google.com/apikey) (free tier)
- _(Phase 2)_ [Neo4j AuraDB Free](https://console.neo4j.io) instance

### 2. Setup

```bash
git clone <repo> && cd semantic-vault

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"

cp .env.example .env
# Edit .env — set GEMINI_API_KEY at minimum
```

### 3. Run the web UI

```bash
python app.py
# Opens http://127.0.0.1:7860
```

### 4. Run the CLI

```bash
# Add knowledge
python main.py ingest "Python was created by Guido van Rossum in 1991."
python main.py ingest --file my_notes.txt --name "Project notes"

# Ask questions
python main.py query "Who created Python?"
python main.py query "Who does Alice work with?" --top-k 5

# List documents
python main.py list

# Re-index existing documents into Neo4j (after connecting it)
python main.py rebuild-graph
python main.py rebuild-graph --doc "specific_doc_name"
```

---

## Feature tiers

### Phase 1 — always active

- Sentence-boundary chunking with overlap
- Gemini extracts entities, relationships, facts, summary, topics per chunk
- Gemini Embedding 2 (3072-dim) dense vectors → Qdrant
- Cosine similarity search with entity-type filtering
- Duplicate detection (SHA-256), cited answers, streaming UI

### Phase 2 — active when `NEO4J_URI` is set

- 3-stage entity resolution (embedding blocking → LLM disambiguation → canonical nodes)
- Knowledge graph in Neo4j: entity nodes, typed `RELATES_TO` edges, bi-temporal provenance
- Conflict detection: `CONTRADICTS` edges when sources disagree
- Relational query routing: Gemini Text2Cypher for graph traversal
- `rebuild-graph` CLI command to retroactively index existing documents

### Phase 3 — active when flags are enabled in `.env`

```bash
ENABLE_QUERY_PLANNER=true   # decomposes multi-part questions into sub-queries
ENABLE_RERANKING=true        # LLM-as-judge reranks retrieval candidates
ENABLE_AUTHORITY=true        # source trust scores weight conflict resolution
```

- Parallel retrieval: dense + graph + structured fire concurrently via `ThreadPoolExecutor`
- RRF fusion: Reciprocal Rank Fusion merges multi-sub-question result lists
- Ontology evolution: `Concept` entity clusters → proposed new type names (background)

---

## Running tests

```bash
# Unit + integration tests — no API key required (all mocked)
pytest tests/ --ignore=tests/test_e2e.py -v          # 138 tests

# End-to-end tests — requires GEMINI_API_KEY in .env
pytest tests/test_e2e.py -v

# Comprehensive manual stretch test
# See: tests/stretch_test.md
```

### Test categories

| File(s) | Phase | What's covered |
|---|---|---|
| `test_chunker.py` | 1 | Sentence splitting, overlap, edge cases |
| `test_extractor.py` | 1+2 | Gemini mock, entity clamping, relationship normalisation |
| `test_storage.py` | 1 | Qdrant upsert, hybrid search, entity-type filter, SQLite CRUD |
| `test_ingestion.py` | 1 | Full pipeline, deduplication, extraction failure tolerance |
| `test_retrieval.py` | 1 | Query→search→synthesis, empty store, top-k |
| `test_entity_resolver.py` | 2 | 3-stage resolution, LLM merge/split, index growth |
| `test_graph_store.py` | 2 | Neo4j driver mocked, Cypher correctness, conflict detection |
| `test_phase2_ingestion.py` | 2 | Graph writes, alias accumulation, fault isolation |
| `test_phase2_retrieval.py` | 2 | Relational query classifier, graph branch, fallback |
| `test_query_planner.py` | 3 | Intent classification, sub-question decomposition, sanitisation |
| `test_reranker.py` | 3 | RRF fusion, LLM reranking, fallback on error |
| `test_authority_scorer.py` | 3 | Score formula, corroboration/contradiction, prompt formatting |
| `test_ontology_engine.py` | 3 | Clustering, type proposal, schema registry, deduplication |
| `test_phase3_retrieval.py` | 3 | Parallel retrieval, Phase 3 backwards-compat with Phase 1 |
| `test_e2e.py` | 1 | Live pipeline with real Gemini API (needs key) |

---

## Project structure

```
semantic-vault/
├── app.py                        Web UI (Gradio 6, two-panel)
├── main.py                       CLI (ingest / query / list / rebuild-graph)
├── pyproject.toml
├── .env.example
│
├── semantic_vault/
│   ├── config.py                 All settings via pydantic-settings
│   ├── models.py                 Pydantic data models (all phases)
│   ├── chunker.py                Sentence-boundary chunker
│   ├── extractor.py              Gemini entity + relationship extraction
│   ├── storage.py                VectorStore (Qdrant) · EntityIndex · MetadataStore (SQLite)
│   ├── ingestion.py              Full ingestion pipeline (Phase 1+2)
│   ├── retrieval.py              Full retrieval pipeline (Phase 1+2+3)
│   ├── entity_resolver.py        3-stage entity resolution
│   ├── graph_store.py            Neo4j wrapper + Text2Cypher
│   ├── query_planner.py          Phase 3 query decomposition
│   ├── reranker.py               RRF fusion + LLM-as-judge reranker
│   ├── authority_scorer.py       Source trust scoring (SQLite)
│   └── ontology_engine.py        Entity clustering + type evolution (SQLite registry)
│
└── tests/
    ├── conftest.py               Shared fixtures (mock embed, mock graph)
    ├── stretch_test.md           Comprehensive manual test corpus + questions
    └── test_*.py                 138 automated tests
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | _(required)_ | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Extraction + synthesis model |
| `GEMINI_SYNTHESIS_MODEL` | `gemini-3-flash-preview` | Synthesis model (can differ) |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-2` | Embedding model |
| `GEMINI_EMBED_DIM` | `3072` | Embedding dimension |
| `QDRANT_URL` | `./qdrant_db` | Qdrant path or `http://...` server |
| `DB_PATH` | `semantic_vault.db` | SQLite path (metadata + schema + authority) |
| `NEO4J_URI` | _(blank)_ | `bolt://...` or `neo4j+s://...` (AuraDB) |
| `NEO4J_USER` | `neo4j` | |
| `NEO4J_PASSWORD` | _(blank)_ | |
| `TOP_K` | `8` | Retrieval results returned |
| `ER_HIGH_THRESHOLD` | `0.85` | Entity resolution: definite match threshold |
| `ER_LOW_THRESHOLD` | `0.65` | Entity resolution: definitely different threshold |
| `ENABLE_QUERY_PLANNER` | `false` | Phase 3: query decomposition |
| `ENABLE_RERANKING` | `false` | Phase 3: LLM-as-judge reranking |
| `ENABLE_AUTHORITY` | `false` | Phase 3: source authority in synthesis |
| `RERANK_TOP_N` | `20` | Candidate pool size for reranker |
