#!/usr/bin/env python3
"""Semantic Vault CLI — Phase 1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semantic_vault.config import settings
from semantic_vault.ingestion import ingest_text
from semantic_vault.retrieval import retrieve_and_synthesize
from semantic_vault.storage import MetadataStore, VectorStore


def _build_stores() -> tuple[VectorStore, MetadataStore]:
    return VectorStore(url=settings.qdrant_url), MetadataStore(db_path=settings.db_path)


def cmd_ingest(args: argparse.Namespace) -> None:
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
        name = args.name or Path(args.file).name
    elif args.text:
        content = args.text
        name = args.name or f"inline_{len(content)}chars"
    else:
        print("Error: supply text as a positional argument or via --file.", file=sys.stderr)
        sys.exit(1)

    vs, ms = _build_stores()
    try:
        doc = ingest_text(content, name, vs, ms)
        print(f"✓ Ingested '{doc.name}' — {doc.chunk_count} chunk(s) stored.")
    except ValueError as exc:
        print(f"Skipped: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    vs, _ = _build_stores()
    result = retrieve_and_synthesize(
        args.question,
        vs,
        top_k=args.top_k,
        entity_type_filter=args.filter_type,
    )
    print(f"\n{result.answer}\n")
    if result.sources:
        print("Sources:")
        for s in result.sources:
            print(f"  [{s['rank']}] {s['doc_name']} (chunk {s['chunk_index']}, score {s['score']:.3f})")
            print(f"       {s['snippet']}")
    if result.entities_mentioned:
        print(f"\nEntities mentioned: {', '.join(result.entities_mentioned[:10])}")


def cmd_list(args: argparse.Namespace) -> None:  # noqa: ARG001
    _, ms = _build_stores()
    docs = ms.list_documents()
    if not docs:
        print("No documents ingested yet.")
        return
    print(f"{'Name':<40} {'Chunks':>6}  {'Ingested'}")
    print("-" * 65)
    for d in docs:
        print(f"{d['name']:<40} {d['chunk_count']:>6}  {d['created_at'][:19]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Vault — LLM-powered knowledge base (Phase 1)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Add a document to the knowledge base")
    p_ingest.add_argument("text", nargs="?", help="Inline text to ingest")
    p_ingest.add_argument("--file", "-f", help="Path to a text file")
    p_ingest.add_argument("--name", "-n", help="Document name (default: derived from source)")

    p_query = sub.add_parser("query", help="Query the knowledge base")
    p_query.add_argument("question", help="Question to answer")
    p_query.add_argument("--top-k", type=int, default=settings.top_k, dest="top_k")
    p_query.add_argument("--filter-type", help="Restrict to chunks containing this entity type")

    sub.add_parser("list", help="List all ingested documents")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "query": cmd_query, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    main()
