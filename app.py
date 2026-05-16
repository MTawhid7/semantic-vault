"""Semantic Vault — Gradio 6 web UI (all phases)."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(".env")

import gradio as gr

from semantic_vault.config import settings
from semantic_vault.ingestion import ingest_text
from semantic_vault.retrieval import (
    _GRAPH_SECTION_TEMPLATE,
    _SYNTHESIS_PROMPT,
    _format_graph_rows,
    _is_relational_query,
    _make_client,
)
from semantic_vault.reranker import llm_rerank, rrf_fuse
from semantic_vault.storage import EntityIndex, MetadataStore, VectorStore

# ---------------------------------------------------------------------------
# Phase 1 — always active
# ---------------------------------------------------------------------------

_vs = VectorStore(url=settings.qdrant_url)
_ms = MetadataStore(db_path=settings.db_path)

# ---------------------------------------------------------------------------
# Phase 2 — active when NEO4J_URI is set
# ---------------------------------------------------------------------------

_graph_store = None
_entity_index = None
_entity_resolver = None

if settings.neo4j_uri:
    try:
        from semantic_vault.entity_resolver import EntityResolver
        from semantic_vault.graph_store import GraphStore

        _graph_store = GraphStore()
        _graph_store.verify_connectivity()
        _graph_store.create_indexes()
        _entity_index = EntityIndex(_vs)
        _entity_resolver = EntityResolver(_entity_index)
        print(f"✓ Neo4j: {settings.neo4j_uri.split('@')[-1]}")
    except Exception as _e:
        print(f"  [warn] Neo4j unavailable: {_e}")
        _graph_store = _entity_index = _entity_resolver = None

# ---------------------------------------------------------------------------
# Phase 3 — active when env flags are set
# ---------------------------------------------------------------------------

_query_planner = None
_authority_scorer = None
_ontology_engine = None

if settings.enable_query_planner:
    try:
        from semantic_vault.query_planner import QueryPlanner
        _query_planner = QueryPlanner()
        print("✓ Query planner: enabled")
    except Exception as _e:
        print(f"  [warn] Query planner init failed: {_e}")

if settings.enable_authority:
    try:
        from semantic_vault.authority_scorer import AuthorityScorer
        _authority_scorer = AuthorityScorer(db_path=settings.db_path)
        print("✓ Authority scorer: enabled")
    except Exception as _e:
        print(f"  [warn] Authority scorer init failed: {_e}")

if _entity_index is not None:
    try:
        from semantic_vault.ontology_engine import OntologyEngine, SchemaRegistry
        _ontology_engine = OntologyEngine(
            _entity_index,
            SchemaRegistry(db_path=settings.db_path),
        )
    except Exception:
        pass

_ingest_count = 0  # tracks ingestions to trigger background ontology evolution

# ---------------------------------------------------------------------------
# Feature status badge
# ---------------------------------------------------------------------------

def _feature_badges() -> list[str]:
    badges = []
    if _graph_store:
        badges.append("🔗 Graph")
    if _query_planner:
        badges.append("🧩 Planner")
    if settings.enable_reranking:
        badges.append("📊 Reranker")
    if _authority_scorer:
        badges.append("🛡️ Authority")
    return badges


_badges = _feature_badges()
_badge_str = " · ".join(_badges) if _badges else "Phase 1 only"
_badge_color = "#2a9d4e" if _badges else "#888"

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _library_md() -> str:
    docs = _ms.list_documents()
    if not docs:
        return "*No documents yet — add some on the left.*"
    return "\n".join(
        f"- **{d['name']}** &nbsp;·&nbsp; {d['chunk_count']} chunk(s) &nbsp;·&nbsp; `{d['created_at'][:10]}`"
        for d in docs
    )


def _format_sources(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = [
        f"**[{i}]** *{h['source_doc_name']}* — {h.get('text','')[:120].replace(chr(10),' ')}…"
        for i, h in enumerate(hits[:5], 1)
    ]
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def add_knowledge(text: str, name: str):
    global _ingest_count
    if not text.strip():
        return "⚠️ Please enter some text first.", _library_md()

    doc_name = name.strip() or f"note ({len(text):,} chars)"
    try:
        doc = ingest_text(
            text, doc_name, _vs, _ms,
            entity_index=_entity_index,
            graph_store=_graph_store,
            entity_resolver=_entity_resolver,
        )
        if _authority_scorer:
            _authority_scorer.record_ingestion(doc_name)

        _ingest_count += 1
        if (_ontology_engine is not None
                and _ingest_count % settings.ontology_evolution_interval == 0):
            import threading
            threading.Thread(target=_ontology_engine.run_evolution, daemon=True).start()

        g = " (+ graph)" if _graph_store else ""
        return f"✅ **'{doc.name}'** added — {doc.chunk_count} chunk(s) indexed{g}.", _library_md()
    except ValueError as exc:
        return f"⚠️ {exc}", _library_md()


def handle_query(message: str, history: list[dict]):
    """Streaming generator — full Phase 1+2+3 retrieval pipeline."""
    message = message.strip()
    if not message:
        yield history + [{"role": "assistant", "content": "Please type a question."}], ""
        return

    client = _make_client()

    # ── Query planning (Phase 3) ──────────────────────────────────────────
    if _query_planner is not None:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        plan = _query_planner.plan(message)
        sub_qs = plan.sub_questions
        strategies = plan.strategies
        do_rerank = plan.needs_reranking or settings.enable_reranking
    else:
        sub_qs = [message]
        strategies = ["dense"]
        if _graph_store and _is_relational_query(message):
            strategies.append("graph")
        do_rerank = settings.enable_reranking

    # ── Retrieval ─────────────────────────────────────────────────────────
    search_k = max(settings.rerank_top_n, settings.top_k * 2) if do_rerank else settings.top_k
    graph_rows: list[dict] = []

    if len(sub_qs) > 1 or len(strategies) > 1:
        # Parallel multi-strategy retrieval
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as
        vector_lists: list[list[dict]] = []
        futures = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            for q in sub_qs:
                if "dense" in strategies:
                    futures[ex.submit(_vs.hybrid_search, q, search_k)] = ("dense", q)
            # Graph only for the original (first) sub-question to avoid N×Gemini calls
            if "graph" in strategies and _graph_store and sub_qs:
                futures[ex.submit(_graph_store.text_to_cypher_query, sub_qs[0], client)] = ("graph", sub_qs[0])
            for f in _as(futures, timeout=30):
                kind, _ = futures[f]
                try:
                    res = f.result()
                    if kind == "dense":
                        vector_lists.append(res)
                    else:
                        graph_rows.extend(res[0])
                except Exception:
                    pass
        hits = rrf_fuse(vector_lists)
    else:
        hits = _vs.hybrid_search(message, top_k=search_k)
        if _graph_store and "graph" in strategies:
            try:
                graph_rows, _ = _graph_store.text_to_cypher_query(message, client)
            except Exception:
                pass

    # ── Reranking (Phase 3) ───────────────────────────────────────────────
    if do_rerank and len(hits) > settings.top_k:
        hits = llm_rerank(message, hits, client, top_k=settings.top_k,
                          max_chunks_to_score=settings.rerank_top_n)
    else:
        hits = hits[:settings.top_k]

    if not hits and not graph_rows:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "No relevant information found. Try adding some documents first."},
        ], ""
        return

    # ── Build prompt ──────────────────────────────────────────────────────
    chunks_block, doc_names_seen = "", []
    for i, h in enumerate(hits, 1):
        dn = h.get("source_doc_name", "unknown")
        chunks_block += f"\n[{i}] Source: {dn}, chunk {h.get('chunk_index','?')}\n{h.get('text','')}\n"
        if dn not in doc_names_seen:
            doc_names_seen.append(dn)

    graph_section = ""
    if graph_rows:
        graph_section = _GRAPH_SECTION_TEMPLATE.format(facts=_format_graph_rows(graph_rows))

    authority_section = ""
    if _authority_scorer and doc_names_seen:
        authority_section = _authority_scorer.format_for_prompt(doc_names_seen)

    prompt = _SYNTHESIS_PROMPT.format(
        chunks=chunks_block or "(none)",
        graph_section=graph_section,
        authority_section=authority_section,
        question=message,
    )

    # ── Stream response ───────────────────────────────────────────────────
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "▌"},
    ]
    yield history, ""

    partial = ""
    for chunk in client.models.generate_content_stream(
        model=settings.gemini_synthesis_model, contents=prompt
    ):
        if chunk.text:
            partial += chunk.text
            history[-1] = {"role": "assistant", "content": partial}
            yield history, ""

    partial += _format_sources(hits)
    history[-1] = {"role": "assistant", "content": partial}
    yield history, ""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

CSS = """
#left-col { border-right: 1px solid var(--border-color-primary); padding-right: 1.25rem; }
#lib-box .prose { font-size: 0.85rem; line-height: 1.5; }
#status { min-height: 2rem; }
footer { display: none !important; }
"""

with gr.Blocks(title="Semantic Vault") as demo:

    gr.HTML(f"""
    <div style="padding:0.75rem 0 0.25rem">
      <h1 style="font-size:1.55rem;font-weight:700;margin:0">🗄️ Semantic Vault</h1>
      <p style="margin:0.15rem 0 0;color:var(--body-text-color-subdued);font-size:0.92rem">
        Add anything you know &mdash; ask anything you need.
      </p>
      <p style="margin:0.25rem 0 0;font-size:0.78rem;color:{_badge_color}">
        Active: {_badge_str}
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=4, elem_id="left-col"):
            gr.Markdown("### ✍️ Add Knowledge")
            text_input = gr.Textbox(
                label="Write or paste text",
                placeholder="Type notes, paste articles, research, meeting summaries…",
                lines=13, max_lines=40,
            )
            with gr.Row():
                name_input = gr.Textbox(
                    label="Document name (optional)",
                    placeholder="e.g. 'Project notes Jan 2025'",
                    scale=4, lines=1, max_lines=1,
                )
                add_btn = gr.Button("➕ Add", variant="primary", scale=1, min_width=80)
            status_out = gr.Markdown(value="*Ready.*", elem_id="status")
            gr.Markdown("---\n### 📚 Library")
            library_out = gr.Markdown(value=_library_md, elem_id="lib-box")

        with gr.Column(scale=6):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(
                value=[], height=530, render_markdown=True,
                placeholder=(
                    "<div style='text-align:center;padding:3rem 1rem;color:#999'>"
                    "<div style='font-size:2rem'>💬</div>"
                    "<div style='margin-top:.5rem'>Ask anything about your knowledge base.</div>"
                    "</div>"
                ),
            )
            with gr.Row():
                query_input = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a question about your notes…",
                    scale=8, lines=1, max_lines=4, autofocus=True,
                )
                ask_btn   = gr.Button("Ask ➤",   variant="primary", scale=1, min_width=75)
                clear_btn = gr.Button("🗑️ Clear", scale=1, min_width=90)

    add_btn.click(fn=add_knowledge, inputs=[text_input, name_input],
                  outputs=[status_out, library_out])
    ask_btn.click(fn=handle_query, inputs=[query_input, chatbot],
                  outputs=[chatbot, query_input])
    query_input.submit(fn=handle_query, inputs=[query_input, chatbot],
                       outputs=[chatbot, query_input])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, query_input])


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
    )
