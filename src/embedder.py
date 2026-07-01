"""
Embedder
────────
Wraps SentenceTransformer to produce dense embeddings.

• Uses functools.lru_cache so the model loads once per process.
  Works in both Streamlit (via st.cache_resource wrapper in ui layer)
  AND FastAPI (plain Python cache) — no Streamlit dependency here.
• Supports batch encoding with a progress bar.
"""

from __future__ import annotations

import functools
from typing import List

import numpy as np


# ── Model loader (process-level cache, framework-agnostic) ───────────────────

@functools.lru_cache(maxsize=4)
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load and cache a SentenceTransformer model (cached per process)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ── Public class ─────────────────────────────────────────────────────────────

class Embedder:
    """Thin wrapper around SentenceTransformer. No framework dependency."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = load_embedder(model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return a float32 numpy array of shape (len(texts), dimension)."""
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via dot product
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string → shape (dimension,)."""
        emb = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb[0].astype(np.float32)