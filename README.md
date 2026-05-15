# Semantic Vault

An intelligent knowledge base that lets you add unstructured text and ask questions about it — no schema design, no manual tagging, no SQL.

Powered by **Gemini** (extraction + embeddings + synthesis) and **Qdrant** (vector search). Phase 1 of a larger system described in [`ROADMAP.md`](ROADMAP.md).

---

## How it works

```
You write text
      ↓
Gemini extracts entities, facts, summary, topics (structured JSON)
      ↓
Gemini embeds the chunk → stored in Qdrant with metadata payload
      ↓
You ask a question
      ↓
Qdrant finds the most relevant chunks (cosine similarity)
      ↓
Gemini synthesises a grounded answer with inline citations
```

---

## Quick start

### 1. Prerequisites

- Python 3.11+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Setup

```bash
git clone <repo>
cd semantic-vault

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[ui,dev]"
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 4. Run the web UI

```bash
python app.py
# Opens http://127.0.0.1:7860 in your browser
```

### 5. Or use the CLI

```bash
python main.py ingest "The Python language was created by Guido van Rossum in 1991."
python main.py ingest --file my_notes.txt --name "Project notes"

python main.py query "Who created Python?"
python main.py list
```

---

## Configuration

All settings are via environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Google AI Studio key |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Extraction model |
| `GEMINI_SYNTHESIS_MODEL` | `gemini-3-flash-preview` | Query synthesis model |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-2` | Embedding model |
| `GEMINI_EMBED_DIM` | `3072` | Output vector dimension |
| `QDRANT_URL` | `./qdrant_db` | Qdrant path or `http://...` server URL |
| `DB_PATH` | `semantic_vault.db` | SQLite metadata database path |
| `CHUNK_SIZE_CHARS` | `1600` | Target chunk size in characters |
| `CHUNK_OVERLAP_CHARS` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `8` | Number of chunks retrieved per query |

---

## Project structure

```
semantic-vault/
├── app.py                    # Gradio web UI
├── main.py                   # CLI (ingest / query / list)
├── semantic_vault/
│   ├── config.py             # pydantic-settings (reads .env)
│   ├── models.py             # Pydantic data models
│   ├── chunker.py            # Sentence-boundary text chunker with overlap
│   ├── extractor.py          # Gemini structured extraction (entities, facts, summary)
│   ├── storage.py            # VectorStore (Qdrant) + MetadataStore (SQLite)
│   ├── ingestion.py          # Orchestration: text → chunks → extract → embed → store
│   └── retrieval.py          # Hybrid search → Gemini synthesis + citations
└── tests/
    ├── conftest.py           # Shared fixtures (in-memory stores, mock embeddings)
    ├── test_chunker.py
    ├── test_extractor.py     # Gemini mocked
    ├── test_storage.py       # In-memory Qdrant + mock embeddings
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── test_e2e.py           # Live API tests (needs GEMINI_API_KEY)
```

---

## Running tests

```bash
# All unit + integration tests (no API key needed — everything is mocked)
pytest tests/ --ignore=tests/test_e2e.py -v

# End-to-end tests against the real Gemini API (needs GEMINI_API_KEY in .env)
pytest tests/test_e2e.py -v
```

---

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the full multi-phase plan.
