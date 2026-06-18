"""Unit tests for the dependency-free text splitter (no network, fully deterministic)."""
from rag.splitter import split_text


def test_empty_input():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_short_text_is_single_chunk():
    assert split_text("hello world foo bar", chunk_size=1000) == ["hello world foo bar"]


def test_long_text_splits_into_bounded_chunks():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = split_text(text, chunk_size=80, chunk_overlap=30)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()) & set(b.split()), "expected shared tokens between chunks"


def test_never_splits_a_word():
    text = "supercalifragilistic " * 10
    for chunk in split_text(text, chunk_size=30, chunk_overlap=5):
        for word in chunk.split():
            assert word == "supercalifragilistic"


def test_giant_word_is_kept_not_dropped():
    giant = "x" * 50
    chunks = split_text(giant, chunk_size=10, chunk_overlap=2)
    assert giant in chunks


def test_overlap_clamped_when_too_large():
    # chunk_overlap >= chunk_size must not loop forever or crash.
    chunks = split_text(" ".join(f"t{i}" for i in range(50)), chunk_size=20, chunk_overlap=999)
    assert len(chunks) >= 1
