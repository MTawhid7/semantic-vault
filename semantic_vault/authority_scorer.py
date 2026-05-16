"""Phase 3 — Source authority scoring.

Tracks how often each document's facts are corroborated or contradicted by other
sources.  Authority scores weight the synthesis prompt so that high-trust sources
are preferred when resolving conflicts.

Backed by the ``source_authority`` SQLite table (added to MetadataStore's DB).
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuthorityScorer:
    """Reads and updates per-document authority scores in SQLite."""

    def __init__(self, db_path: str = "semantic_vault.db") -> None:
        self._path = db_path
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_authority (
                    doc_name            TEXT PRIMARY KEY,
                    corroboration_count INTEGER NOT NULL DEFAULT 0,
                    contradiction_count INTEGER NOT NULL DEFAULT 0,
                    authority_score     REAL    NOT NULL DEFAULT 0.5,
                    last_updated        TEXT    NOT NULL
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_ingestion(self, doc_name: str) -> None:
        """Register a new document with default authority 0.5."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO source_authority
                    (doc_name, corroboration_count, contradiction_count,
                     authority_score, last_updated)
                VALUES (?, 0, 0, 0.5, ?)
                """,
                (doc_name, _now_iso()),
            )
            conn.commit()

    def record_corroboration(self, doc_name: str, count: int = 1) -> None:
        """Increment corroboration count and recompute authority."""
        self._update_counts(doc_name, corroboration_delta=count)

    def record_contradiction(self, doc_name: str, count: int = 1) -> None:
        """Increment contradiction count and recompute authority."""
        self._update_counts(doc_name, contradiction_delta=count)

    def _update_counts(
        self,
        doc_name: str,
        corroboration_delta: int = 0,
        contradiction_delta: int = 0,
    ) -> None:
        self.record_ingestion(doc_name)  # ensure row exists
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE source_authority
                SET corroboration_count = corroboration_count + ?,
                    contradiction_count = contradiction_count + ?,
                    last_updated = ?
                WHERE doc_name = ?
                """,
                (corroboration_delta, contradiction_delta, _now_iso(), doc_name),
            )
            conn.commit()
        self._recompute(doc_name)

    def _recompute(self, doc_name: str) -> None:
        """authority = corroborations / (corroborations + contradictions + 1)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT corroboration_count, contradiction_count FROM source_authority WHERE doc_name = ?",
                (doc_name,),
            ).fetchone()
            if row is None:
                return
            c, d = row["corroboration_count"], row["contradiction_count"]
            score = c / (c + d + 1)
            conn.execute(
                "UPDATE source_authority SET authority_score = ? WHERE doc_name = ?",
                (round(score, 4), doc_name),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_score(self, doc_name: str) -> float:
        """Return authority score in [0, 1]. Defaults to 0.5 for unknown sources."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT authority_score FROM source_authority WHERE doc_name = ?",
                (doc_name,),
            ).fetchone()
            return float(row["authority_score"]) if row else 0.5

    def get_all_scores(self) -> dict[str, float]:
        """Return {doc_name: authority_score} for all tracked documents."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_name, authority_score FROM source_authority"
            ).fetchall()
            return {r["doc_name"]: float(r["authority_score"]) for r in rows}

    def format_for_prompt(self, doc_names: list[str]) -> str:
        """Return a short authority summary string to include in the synthesis prompt."""
        scores = self.get_all_scores()
        lines = []
        for name in doc_names:
            score = scores.get(name, 0.5)
            tier = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"
            lines.append(f"  {name}: trust={tier} ({score:.2f})")
        return "Source authority:\n" + "\n".join(lines) if lines else ""
