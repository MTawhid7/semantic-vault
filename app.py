"""Semantic Vault — Gradio 6 web interface."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # must run before importing semantic_vault so settings picks up .env

import gradio as gr

from semantic_vault.config import settings
from semantic_vault.ingestion import ingest_text
from semantic_vault.retrieval import _SYNTHESIS_PROMPT, _make_client
from semantic_vault.storage import MetadataStore, VectorStore

# ---------------------------------------------------------------------------
# Persistent stores (live for the entire process lifetime)
# ---------------------------------------------------------------------------

_vs = VectorStore(url=settings.qdrant_url)
_ms = MetadataStore(db_path=settings.db_path)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _library_md() -> str:
    docs = _ms.list_documents()
    if not docs:
        return "*No documents yet — add some on the left.*"
    lines = [
        f"- **{d['name']}** &nbsp;·&nbsp; {d['chunk_count']} chunk(s) &nbsp;·&nbsp; `{d['created_at'][:10]}`"
        for d in docs
    ]
    return "\n".join(lines)


def _format_sources(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = []
    for i, h in enumerate(hits[:5], start=1):
        snippet = h.get("text", "")[:120].replace("\n", " ")
        lines.append(f"**[{i}]** *{h['source_doc_name']}* — {snippet}…")
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def add_knowledge(text: str, name: str):
    """Ingest a document. Returns (status_markdown, library_markdown)."""
    if not text.strip():
        return "⚠️ Please enter some text first.", _library_md()

    doc_name = name.strip() or f"note ({len(text):,} chars)"
    try:
        doc = ingest_text(text, doc_name, _vs, _ms)
        status = f"✅ **'{doc.name}'** added — {doc.chunk_count} chunk(s) indexed."
        return status, _library_md()
    except ValueError as exc:
        return f"⚠️ {exc}", _library_md()


def handle_query(message: str, history: list[dict]):
    """Streaming generator — retrieves chunks then streams Gemini's answer."""
    message = message.strip()
    if not message:
        yield history + [{"role": "assistant", "content": "Please type a question."}], ""
        return

    hits = _vs.hybrid_search(message, top_k=settings.top_k)

    if not hits:
        no_info = "No relevant information found in the knowledge base. Try adding some documents first."
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": no_info},
        ], ""
        return

    # Build context block
    chunks_block = ""
    for i, h in enumerate(hits, start=1):
        chunks_block += f"\n[{i}] Source: {h['source_doc_name']}, chunk {h.get('chunk_index', '?')}\n{h['text']}\n"

    prompt = _SYNTHESIS_PROMPT.format(chunks=chunks_block, question=message)
    client = _make_client()

    # Append user message + empty assistant placeholder immediately
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "▌"},
    ]
    yield history, ""  # clears input, shows user message right away

    # Stream tokens
    partial = ""
    stream = client.models.generate_content_stream(
        model=settings.gemini_synthesis_model,
        contents=prompt,
    )
    for chunk in stream:
        if chunk.text:
            partial += chunk.text
            history[-1] = {"role": "assistant", "content": partial}
            yield history, ""

    # Append sources after streaming finishes
    partial += _format_sources(hits)
    history[-1] = {"role": "assistant", "content": partial}
    yield history, ""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
#left-col  { border-right: 1px solid var(--border-color-primary); padding-right: 1.25rem; }
#lib-box .prose { font-size: 0.85rem; line-height: 1.5; }
#status    { min-height: 2rem; }
footer     { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Semantic Vault") as demo:

    gr.HTML("""
    <div style="padding:0.75rem 0 0.25rem">
      <h1 style="font-size:1.55rem;font-weight:700;margin:0">🗄️ Semantic Vault</h1>
      <p style="margin:0.2rem 0 0;color:var(--body-text-color-subdued);font-size:0.92rem">
        Add anything you know &mdash; ask anything you need.
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left: Add knowledge ────────────────────────────────────────────
        with gr.Column(scale=4, elem_id="left-col"):
            gr.Markdown("### ✍️ Add Knowledge")

            text_input = gr.Textbox(
                label="Write or paste text",
                placeholder=(
                    "Type your notes, paste articles, research, meeting summaries…\n\n"
                    "Anything added here becomes searchable and queryable on the right."
                ),
                lines=13,
                max_lines=40,
            )
            with gr.Row():
                name_input = gr.Textbox(
                    label="Document name (optional)",
                    placeholder="e.g. 'Meeting notes Jan 2025'",
                    scale=4,
                    lines=1,
                    max_lines=1,
                )
                add_btn = gr.Button("➕ Add", variant="primary", scale=1, min_width=80)

            status_out = gr.Markdown(value="*Ready.*", elem_id="status")

            gr.Markdown("---\n### 📚 Library")
            library_out = gr.Markdown(value=_library_md, elem_id="lib-box")

        # ── Right: Chat / query ────────────────────────────────────────────
        with gr.Column(scale=6):
            gr.Markdown("### 💬 Ask Questions")

            chatbot = gr.Chatbot(
                value=[],
                height=530,
                render_markdown=True,
                placeholder=(
                    "<div style='text-align:center;padding:3rem 1rem;color:#999'>"
                    "<div style='font-size:2rem'>💬</div>"
                    "<div style='margin-top:.5rem'>Ask anything about your knowledge base.</div>"
                    "<div style='font-size:.85rem;margin-top:.35rem'>Answers are grounded in what you've added.</div>"
                    "</div>"
                ),
            )

            with gr.Row():
                query_input = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a question about your notes…",
                    scale=8,
                    lines=1,
                    max_lines=4,
                    autofocus=True,
                )
                ask_btn  = gr.Button("Ask ➤",  variant="primary", scale=1, min_width=75)
                clear_btn = gr.Button("🗑️ Clear", scale=1, min_width=90)

    # ── Events ────────────────────────────────────────────────────────────

    add_btn.click(
        fn=add_knowledge,
        inputs=[text_input, name_input],
        outputs=[status_out, library_out],
    )

    ask_btn.click(
        fn=handle_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input],
    )
    query_input.submit(
        fn=handle_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input],
    )
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, query_input])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
    )
