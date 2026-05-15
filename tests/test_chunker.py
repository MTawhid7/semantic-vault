"""Unit tests for the text chunker."""
import pytest

from semantic_vault.chunker import chunk_text


def test_empty_text_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk():
    text = "Hello world. This is a short sentence."
    chunks = chunk_text(text, chunk_size=1600)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0]


def test_long_text_splits_into_multiple_chunks():
    # 10 sentences, each ~100 chars, chunk_size=300 → ~3-4 chunks
    sentences = [f"This is sentence number {i} and it has some padding text here." for i in range(10)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1


def test_chunks_cover_all_content():
    """Every sentence should appear in at least one chunk."""
    sentences = [f"Unique marker sentence {i}." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    full = " ".join(chunks)
    for i in range(20):
        assert f"Unique marker sentence {i}" in full


def test_overlap_carries_sentences_forward():
    """The last sentence(s) of chunk N should appear in chunk N+1."""
    # Make chunks small enough to force a split
    sentences = [f"Sentence {c} here." for c in "ABCDEFGHIJ"]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=80, overlap=40)
    if len(chunks) >= 2:
        # Last word(s) of chunk 0 should appear somewhere in chunk 1
        last_word_chunk0 = chunks[0].split()[-1]
        assert last_word_chunk0 in chunks[1]


def test_single_very_long_sentence_becomes_one_chunk():
    long_sentence = "word " * 500  # no sentence-ending punctuation
    chunks = chunk_text(long_sentence, chunk_size=200, overlap=50)
    # Can't split without punctuation, so we get one chunk (the whole text)
    assert len(chunks) == 1


def test_paragraph_breaks_are_respected():
    text = "First paragraph sentence one. First paragraph sentence two.\n\nSecond paragraph sentence one."
    chunks = chunk_text(text, chunk_size=1600)
    assert len(chunks) == 1  # small enough to fit in one chunk
    assert "Second paragraph" in chunks[0]


def test_no_empty_chunks():
    text = "A. B. C. D. E. F."
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    for chunk in chunks:
        assert chunk.strip() != ""
