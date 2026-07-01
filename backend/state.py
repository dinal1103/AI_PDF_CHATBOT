"""
backend/state.py
────────────────
Single in-memory state object for the FastAPI backend.

Holds the embedder and vector store so they survive across requests.
In production you'd replace this with Redis + a persistent vector DB.
For a portfolio / small-team tool, in-memory is fine.
"""

from __future__ import annotations

from typing import Optional

from src.document_processor import DocumentInfo, process_document
from src.embedder import Embedder
from src.vector_store import FAISSVectorStore
from src.rag_chain import answer_with_history


class AppState:
    """
    Singleton-style container for all ML objects.
    One instance is created in routes.py and reused for every request.
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self._embedder: Optional[Embedder] = None
        self._store: Optional[FAISSVectorStore] = None
        self._doc_info: Optional[DocumentInfo] = None

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest(self, file_obj, chunk_size: int = 500, chunk_overlap: int = 50) -> DocumentInfo:
        """
        Process a file, embed all chunks, build FAISS index.
        Replaces any existing document.
        """
        chunks, doc_info = process_document(file_obj, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if self._embedder is None:
            self._embedder = Embedder(self.EMBED_MODEL)

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        store = FAISSVectorStore(dimension=self._embedder.dimension)
        store.add(
            embeddings,
            texts,
            sources=[c.source for c in chunks],
            pages=[c.page for c in chunks],
        )

        self._store = store
        self._doc_info = doc_info
        return doc_info

    # ── Query ────────────────────────────────────────────────────────────────

    def answer(self, query: str, history: list, groq_api_key: str, **kwargs):
        """Run the RAG chain. Returns (generator_or_str, list[RetrievalResult])."""
        if not self.is_ready():
            raise RuntimeError("No document loaded.")
        return answer_with_history(
            query=query,
            history=history,
            vector_store=self._store,
            embedder=self._embedder,
            groq_api_key=groq_api_key,
            **kwargs,
        )

    # ── Utilities ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._store is not None and self._store.total_vectors > 0

    def get_info(self) -> dict:
        if self._doc_info is None:
            return {"doc_name": None, "total_chunks": 0, "embed_model": self.EMBED_MODEL}
        return {
            "doc_name": self._doc_info.name,
            "total_chunks": self._doc_info.total_chunks,
            "embed_model": self.EMBED_MODEL,
        }

    def reset(self):
        """Clear everything. Next upload starts fresh."""
        self._store = None
        self._doc_info = None
        # Keep embedder cached — model reload is expensive