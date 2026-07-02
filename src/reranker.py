"""
src/reranker.py
───────────────
Cross-encoder re-ranking stage applied AFTER FAISS retrieval.
Bi-encoder (embedder.py) retrieval is fast but approximate; the
cross-encoder scores (query, chunk) pairs jointly for much higher
precision on the top candidates.

FIX: Added graceful fallback — if the cross-encoder model cannot be
downloaded (no internet, cold start timeout, or memory limit on free
hosting), reranker silently skips and returns original FAISS order.
This prevents the first question from crashing on Render free tier.
"""

from __future__ import annotations
import functools
import logging
from typing import List
from .vector_store import RetrievalResult

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._available = False
        try:
            _load_cross_encoder(model_name)
            self._available = True
        except Exception as e:
            # FIX: don't crash on import — log and fall back to FAISS scores
            logger.warning(f"Reranker unavailable (will use FAISS scores): {e}")

    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int = 5
    ) -> List[RetrievalResult]:
        if not results:
            return results

        # FIX: if model failed to load, just return top_k by FAISS score
        if not self._available:
            return results[:top_k]

        try:
            model = _load_cross_encoder(self.model_name)
            pairs = [(query, r.text) for r in results]
            scores = model.predict(pairs)
            rescored = [
                RetrievalResult(
                    chunk_id=r.chunk_id, text=r.text,
                    score=float(s), source=r.source, page=r.page,
                )
                for r, s in zip(results, scores)
            ]
            rescored.sort(key=lambda r: r.score, reverse=True)
            return rescored[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed, using FAISS order: {e}")
            return results[:top_k]