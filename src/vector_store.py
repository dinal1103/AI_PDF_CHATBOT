"""
Vector Store
────────────
FAISS-based in-memory vector store.

Design notes
------------
• Uses IndexFlatIP (inner product) on L2-normalised vectors → cosine similarity.
• Stores chunk texts + metadata in a parallel list; no external DB required.
• Optional MMR (Maximal Marginal Relevance) re-ranking to reduce redundancy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss


# ── Data class for a retrieval result ───────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id: int
    text: str
    score: float
    source: str
    page: Optional[int] = None


# ── Vector store ─────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Lightweight FAISS wrapper that also stores chunk texts for retrieval.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # cosine via normalised vecs
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._pages: List[Optional[int]] = []

    # ── Ingestion ────────────────────────────────────────────────────────────

    def add(
        self,
        embeddings: np.ndarray,       # (n, dim) float32
        texts: List[str],
        sources: Optional[List[str]] = None,
        pages: Optional[List[Optional[int]]] = None,
    ) -> None:
        """Add a batch of embeddings + metadata to the store."""
        assert embeddings.shape[0] == len(texts), "embeddings / texts length mismatch"
        self.index.add(embeddings)
        self._texts.extend(texts)
        self._sources.extend(sources or [""] * len(texts))
        self._pages.extend(pages or [None] * len(texts))

    # ── Retrieval ────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,   # (dim,) float32
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Nearest-neighbour search. Returns up to top_k results."""
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding[np.newaxis, :], k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=int(idx),
                    text=self._texts[idx],
                    score=float(score),
                    source=self._sources[idx],
                    page=self._pages[idx],
                )
            )
        return results

    def search_mmr(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
    ) -> List[RetrievalResult]:
        """
        Maximal Marginal Relevance re-ranking.

        Retrieves `fetch_k` candidates then iteratively picks the result
        that maximises (lambda_mult * relevance) - ((1 - lambda_mult) * redundancy).
        """
        candidates = self.search(query_embedding, top_k=min(fetch_k, self.index.ntotal))
        if not candidates:
            return []

        cand_embeds = np.array(
            [self._get_embedding(c.chunk_id) for c in candidates], dtype=np.float32
        )

        selected_indices: List[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_indices) < top_k and remaining:
            if not selected_indices:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: candidates[i].score)
            else:
                sel_embeds = cand_embeds[selected_indices]
                best_score = -np.inf
                best = remaining[0]
                for i in remaining:
                    rel = candidates[i].score
                    # max similarity to already-selected
                    sim = float(np.max(cand_embeds[i] @ sel_embeds.T))
                    mmr_score = lambda_mult * rel - (1 - lambda_mult) * sim
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i
            selected_indices.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected_indices]

    def _get_embedding(self, idx: int) -> np.ndarray:
        """Reconstruct a stored embedding from the flat FAISS index."""
        emb = np.zeros(self.dimension, dtype=np.float32)
        self.index.reconstruct(idx, emb)
        return emb

    # ── Utility ──────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        """Clear all vectors and metadata."""
        self.index.reset()
        self._texts.clear()
        self._sources.clear()
        self._pages.clear()