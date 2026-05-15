"""Split documents into overlapping chunks on sentence boundaries."""
from __future__ import annotations

import re


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    # Split on sentence-ending punctuation followed by whitespace or newline
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Also split on double newlines (paragraph breaks)
    sentences: list[str] = []
    for part in parts:
        subparts = re.split(r"\n\n+", part)
        sentences.extend(s.strip() for s in subparts if s.strip())
    return sentences


def chunk_text(
    text: str,
    chunk_size: int = 1600,
    overlap: int = 200,
) -> list[str]:
    """Return overlapping text chunks, splitting on sentence boundaries.

    Each chunk is at most ``chunk_size`` characters. Consecutive chunks share
    the last ``overlap`` characters worth of sentences from the previous chunk.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence)

        if current_len + slen > chunk_size and current:
            chunks.append(" ".join(current))

            # Carry forward overlap worth of sentences
            overlap_buf: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= overlap:
                    overlap_buf.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_buf
            current_len = overlap_len

        current.append(sentence)
        current_len += slen

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text.strip()]
