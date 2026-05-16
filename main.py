#!/usr/bin/env python3
"""Semantic Vault CLI — Phase 1 + Phase 2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

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

    graph_store = None
    if settings.neo4j_uri:
        try:
            from semantic_vault.graph_store import GraphStore
            graph_store = GraphStore()
        except Exception as exc:
            print(f"  [warn] graph unavailable: {exc}", file=sys.stderr)

    result = retrieve_and_synthesize(
        args.question,
        vs,
        top_k=args.top_k,
        entity_type_filter=args.filter_type,
        graph_store=graph_store,
    )
    print(f"\n{result.answer}\n")
    if result.graph_facts:
        print(f"Graph facts used: {len(result.graph_facts)}")
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


def cmd_rebuild_graph(args: argparse.Namespace) -> None:
    """Re-index all existing documents into the Neo4j knowledge graph.

    Uses extraction data cached in the Qdrant payload when available.
    Falls back to re-extracting with Gemini for documents ingested before
    Phase 2 (which don't have relationship data in their payloads).
    """
    if not settings.neo4j_uri:
        print("Error: NEO4J_URI is not set in .env — nothing to rebuild.", file=sys.stderr)
        sys.exit(1)

    from semantic_vault.entity_resolver import EntityResolver
    from semantic_vault.extractor import extract
    from semantic_vault.graph_store import GraphStore
    from semantic_vault.ingestion import _process_graph
    from semantic_vault.models import (
        Chunk,
        ExtractionOutput,
        ExtractedEntity,
        ExtractedRelationship,
    )
    from semantic_vault.storage import EntityIndex

    vs, ms = _build_stores()
    docs = ms.list_documents()

    if not docs:
        print("No documents in the knowledge base.")
        return

    print("Connecting to Neo4j…")
    graph_store = GraphStore()
    graph_store.verify_connectivity()
    graph_store.create_indexes()
    entity_index = EntityIndex(vs)
    resolver = EntityResolver(entity_index)

    print(f"Re-indexing {len(docs)} document(s) into the knowledge graph…\n")

    total_entities = 0
    total_rels = 0

    for doc in docs:
        if args.doc and doc["name"] != args.doc:
            continue

        print(f"  [{doc['name']}]  ({doc['chunk_count']} chunk(s))")
        chunks = vs.get_chunks_by_doc_id(doc["id"])

        for chunk_data in chunks:
            cidx = chunk_data.get("chunk_index", "?")
            text = chunk_data.get("text", "")

            # Fast path: full extraction already stored in payload (Phase 2 ingestions)
            stored_entities = chunk_data.get("entities", [])
            stored_rels = chunk_data.get("relationships", [])

            if stored_entities:
                try:
                    extraction = ExtractionOutput(
                        entities=[ExtractedEntity(**e) for e in stored_entities],
                        relationships=[ExtractedRelationship(**r) for r in stored_rels],
                        key_facts=chunk_data.get("key_facts", []),
                        summary=chunk_data.get("summary", ""),
                        topics=chunk_data.get("topics", []),
                    )
                    print(f"    chunk {cidx}: using cached extraction data")
                except Exception as exc:
                    print(f"    chunk {cidx}: cached data malformed ({exc}), re-extracting…")
                    stored_entities = []

            if not stored_entities:
                # Slow path: re-extract with Gemini (pre-Phase-2 documents)
                print(f"    chunk {cidx}: re-extracting with Gemini…")
                try:
                    extraction = extract(text)
                except Exception as exc:
                    print(f"    chunk {cidx}: extraction failed — {exc}")
                    continue

            chunk = Chunk(
                id=chunk_data["id"],
                text=text,
                source_doc_id=doc["id"],
                source_doc_name=doc["name"],
                chunk_index=int(cidx) if str(cidx).isdigit() else 0,
                extraction=extraction,
            )
            _process_graph(chunk, resolver, graph_store)
            total_entities += len(extraction.entities)
            total_rels += len(extraction.relationships)

    e_count = graph_store.entity_count()
    r_count = graph_store.relationship_count()
    print(
        f"\n✓ Rebuild complete.\n"
        f"  Processed : {total_entities} entity mentions, {total_rels} relationship mentions\n"
        f"  Graph now : {e_count} canonical entities, {r_count} relationships"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Vault — LLM-powered knowledge base"
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

    p_rebuild = sub.add_parser(
        "rebuild-graph",
        help="Re-index existing documents into the Neo4j knowledge graph",
    )
    p_rebuild.add_argument(
        "--doc", help="Rebuild only this document (by name). Default: all documents."
    )

    args = parser.parse_args()
    {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "list": cmd_list,
        "rebuild-graph": cmd_rebuild_graph,
    }[args.command](args)


if __name__ == "__main__":
    main()
