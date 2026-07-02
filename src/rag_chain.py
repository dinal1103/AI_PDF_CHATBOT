"""
src/rag_chain.py
────────────────
Orchestrates retrieval → prompt construction → LLM call → answer.
Framework-agnostic: works in Streamlit, FastAPI, CLI, or tests.
"""

from __future__ import annotations

import functools
import textwrap
from typing import Generator, List, Tuple, Union

from .vector_store import FAISSVectorStore, RetrievalResult
from .embedder import Embedder
from .reranker import Reranker


SYSTEM_PROMPT = textwrap.dedent("""
    You are DocuMind AI, an expert document analyst and question-answering assistant.
    Your job is to answer questions accurately and concisely using ONLY the context
    excerpts provided below.

    Rules:
    1. Answer ONLY from the provided context. Never fabricate information.
    2. If the answer is not found in the context, say:
       "I couldn't find relevant information in the document for this question."
    3. Quote short, key phrases directly from the document when helpful.
    4. Structure long answers with bullet points or numbered lists.
    5. Be concise yet complete. Avoid repeating yourself.
    6. If the question is ambiguous, ask for clarification.
""").strip()


def _build_context_block(results: List[RetrievalResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        page_str = f" | page={r.page}" if r.page else ""
        header = f"[Excerpt {i} | score={r.score:.3f}{page_str}]"
        blocks.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def _build_user_message(query: str, context: str) -> str:
    return (
        f"Context excerpts from the document:\n\n{context}"
        f"\n\n---\n\nUser question: {query}"
    )


def _retrieve(vector_store, embedder, query, top_k, use_mmr, use_rerank=True):
    query_emb = embedder.embed_query(query)
    fetch_k = top_k * 4 if use_rerank else top_k
    if use_mmr:
        candidates = vector_store.search_mmr(query_emb, top_k=fetch_k, fetch_k=fetch_k * 3)
    else:
        candidates = vector_store.search(query_emb, top_k=fetch_k)
    if use_rerank and candidates:
        candidates = _get_reranker().rerank(query, candidates, top_k=top_k)
    else:
        candidates = candidates[:top_k]
    return candidates


@functools.lru_cache(maxsize=8)
def _get_groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)


@functools.lru_cache(maxsize=1)
def _get_reranker():
    return Reranker()


def answer_with_history(
    query: str,
    history: list[dict],
    vector_store: FAISSVectorStore,
    embedder: Embedder,
    groq_api_key: str,
    model: str = "llama-3.3-70b-versatile",
    top_k: int = 5,
    temperature: float = 0.2,
    use_mmr: bool = True,
    stream: bool = True,
) -> Tuple[Union[Generator, str], List[RetrievalResult]]:
    """
    RAG answer with conversation history.
    stream=True  → token generator (Streamlit / SSE)
    stream=False → full string (JSON API response)
    Called by: backend/state.py → backend/api/routes.py
    """
    results = _retrieve(vector_store, embedder, query, top_k, use_mmr, use_rerank=True)

    if not results:
        no_doc = "No document indexed. Please upload a document first."
        if stream:
            def _no_doc_gen():
                yield no_doc
            return _no_doc_gen(), []
        return no_doc, []

    context = _build_context_block(results)
    user_msg = _build_user_message(query, context)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_msg})

    client = _get_groq_client(groq_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        stream=stream,
    )

    if stream:
        def token_generator():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return token_generator(), results
    else:
        return response.choices[0].message.content, results