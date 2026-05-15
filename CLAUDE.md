# CLAUDE.md — Semantic Vault

## Overview

Semantic Vault is a **Python-only** LLM-powered knowledge management system. There is no frontend framework — the UI is Gradio. There is no traditional ORM — storage is Qdrant (vectors) + SQLite (metadata). All LLM calls use the `google-genai` SDK.

---

## Running the project

```bash
# Activate the virtual environment first — always
source .venv/bin/activate

# Web UI
python app.py

# CLI
python main.py ingest "some text"
python main.py query "a question"
python main.py list

# Tests (no API key needed)
pytest tests/ --ignore=tests/test_e2e.py -v

# Live e2e tests (needs GEMINI_API_KEY in .env)
pytest tests/test_e2e.py -v
```

## Installing dependencies

```bash
# Install all (core + UI + dev)
pip install -e ".[ui,dev]"

# Core only (no Gradio, no pytest)
pip install -e .
```

---

## Architecture

### Data flow — ingestion
```
ingest_text()
  → chunker.chunk_text()          # sentence-boundary splitter with overlap
  → extractor.extract()           # Gemini structured JSON: entities, facts, summary, topics
  → VectorStore.embed()           # Gemini embedding-2 → 3072-dim vector
  → VectorStore.upsert_chunks()   # Qdrant point: vector + rich payload
  → MetadataStore.save_document() # SQLite row: id, name, hash, date, chunk_count
```

### Data flow — retrieval
```
retrieve_and_synthesize()
  → VectorStore.embed([query])    # same Gemini embedding model
  → VectorStore.hybrid_search()   # Qdrant cosine similarity search
  → Gemini generate_content()     # synthesis with retrieved chunks as context
  → QueryResult                   # answer + sources list + entity names
```

### Key files

| File | Responsibility |
|---|---|
| `semantic_vault/config.py` | All settings via pydantic-settings; reads `.env` |
| `semantic_vault/models.py` | `ExtractionOutput`, `Chunk`, `SourceDocument`, `QueryResult` |
| `semantic_vault/chunker.py` | `chunk_text()` — sentence splitter, no external deps |
| `semantic_vault/extractor.py` | `extract()` — Gemini tool-call with `response_schema=ExtractionOutput` |
| `semantic_vault/storage.py` | `VectorStore` + `MetadataStore` |
| `semantic_vault/ingestion.py` | `ingest_text()` — orchestrates the whole pipeline |
| `semantic_vault/retrieval.py` | `retrieve_and_synthesize()` + streaming via `generate_content_stream` |
| `app.py` | Gradio Blocks UI: two-column layout (add left, chat right) |
| `main.py` | argparse CLI: `ingest`, `query`, `list` subcommands |

---

## Testing conventions

- **No real API calls in unit/integration tests.** The `vector_store` fixture in `conftest.py` injects `embed_fn=_mock_embed` — a pure-Python deterministic function. Gemini clients are mocked with `MagicMock`.
- **`test_e2e.py`** hits the real API; all tests are decorated with `@_NEEDS_KEY` and skip automatically if `GEMINI_API_KEY` is not set.
- Test structure mirrors the source: one test file per source module.

---

## Important constraints

- **Embedding dimension must match the Qdrant collection.** If `GEMINI_EMBED_DIM` changes in `.env`, delete the `qdrant_db/` folder — `VectorStore._ensure_collection()` will detect the mismatch and raise a clear error.
- **Do not import `fastembed` or `onnxruntime`.** These crash on Python 3.14. Embeddings go through the Gemini API only.
- **Extraction is best-effort.** If `extractor.extract()` raises, `ingest_text()` logs a warning and stores the chunk without extraction metadata. Don't make extraction mandatory.
- **Entity types are clamped.** After Gemini extraction, `_clamp_entity_types()` normalises unknown types to `"Concept"`. The base ontology is in `models.BASE_ENTITY_TYPES`.

---

## Phase 2 / Phase 3 notes

See `ROADMAP.md`. The next phase adds **Neo4j** for a knowledge graph and **entity resolution**. The storage module will grow a `GraphStore` class alongside `VectorStore`. The ingestion pipeline will gain a 3-stage entity resolution step before graph writes.
