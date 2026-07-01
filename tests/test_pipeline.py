"""
Unit tests for DocuMind AI core pipeline.
Run with:  pytest tests/ -v
"""

import io
import numpy as np
import pytest

from src.document_processor import SemanticChunker, _clean_text
from src.vector_store import FAISSVectorStore


# ── Text cleaning ─────────────────────────────────────────────────────────────

def test_clean_text_strips_control_chars():
    dirty = "Hello\x00World\x1fTest"
    assert "\x00" not in _clean_text(dirty)
    assert "Hello" in _clean_text(dirty)


def test_clean_text_collapses_blank_lines():
    text = "A\n\n\n\nB"
    assert _clean_text(text) == "A\n\nB"


# ── Chunker ───────────────────────────────────────────────────────────────────

def test_chunker_produces_chunks():
    text = "This is sentence one. This is sentence two. This is sentence three. " * 10
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split(text, source="test.txt")
    assert len(chunks) > 1, "Expected multiple chunks for long text"


def test_chunker_chunk_ids_sequential():
    text = " ".join(["Word"] * 500)
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.split(text)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(ids))), "Chunk IDs should be sequential"


def test_chunker_short_text_single_chunk():
    text = "Short text."
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split(text)
    assert len(chunks) == 1


def test_chunk_metadata():
    text = "Hello world. Foo bar."
    chunker = SemanticChunker(chunk_size=500)
    chunks = chunker.split(text, source="doc.pdf")
    assert chunks[0].source == "doc.pdf"
    assert chunks[0].word_count > 0
    assert chunks[0].char_count > 0


# ── Vector store ──────────────────────────────────────────────────────────────

def test_faiss_add_and_search():
    dim = 8
    store = FAISSVectorStore(dimension=dim)
    
    # Create random normalised vectors
    rng = np.random.default_rng(42)
    vecs = rng.random((5, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms

    texts = [f"chunk {i}" for i in range(5)]
    store.add(vecs, texts)

    assert store.total_vectors == 5

    query = vecs[0].copy()
    results = store.search(query, top_k=3)
    assert len(results) == 3
    # Top result should be the same vector (score ≈ 1.0)
    assert results[0].score > 0.99


def test_faiss_mmr():
    dim = 8
    store = FAISSVectorStore(dimension=dim)
    rng = np.random.default_rng(0)
    vecs = rng.random((10, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    texts = [f"chunk {i}" for i in range(10)]
    store.add(vecs, texts)

    query = vecs[0]
    results = store.search_mmr(query, top_k=4, fetch_k=10)
    assert len(results) == 4
    # No duplicate chunk IDs
    ids = [r.chunk_id for r in results]
    assert len(ids) == len(set(ids))


def test_faiss_reset():
    dim = 4
    store = FAISSVectorStore(dimension=dim)
    vecs = np.ones((3, dim), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add(vecs, ["a", "b", "c"])
    store.reset()
    assert store.total_vectors == 0


def test_faiss_empty_search():
    store = FAISSVectorStore(dimension=4)
    query = np.zeros(4, dtype=np.float32)
    results = store.search(query, top_k=5)
    assert results == []