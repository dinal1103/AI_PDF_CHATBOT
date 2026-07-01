"""
Reranker
────────
Cross-encoder re-ranking stage applied AFTER FAISS retrieval.
Bi-encoder (embedder.py) retrieval is fast but approximate; the
cross-encoder scores (query, chunk) pairs jointly for much higher
precision on the top candidates. Standard two-stage retrieval design.
"""

from __future__ import annotations
import functools
from typing import List
from .vector_store import RetrievalResult


@functools.lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = _load_cross_encoder(model_name)

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        if not results:
            return results
        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs)
        rescored = [
            RetrievalResult(chunk_id=r.chunk_id, text=r.text, score=float(s), source=r.source, page=r.page)
            for r, s in zip(results, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]