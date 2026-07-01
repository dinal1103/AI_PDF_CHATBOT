"""
RAG Chain
─────────
Orchestrates retrieval → prompt construction → LLM call → answer.

Flow
----
1. Embed the user query.
2. Retrieve top-k chunks from FAISSVectorStore (with optional MMR).
3. Build a structured prompt with the retrieved context.
4. Call Groq LLaMA-3 via the Groq SDK.
5. Return the answer + source chunks.

Framework-agnostic: works in Streamlit, FastAPI, CLI, or tests.
Groq client cached via functools.lru_cache — no Streamlit dependency.
"""

from __future__ import annotations

import functools
import textwrap
from typing import Generator, List, Tuple, Union

from .vector_store import FAISSVectorStore, RetrievalResult
from .embedder import Embedder
from .reranker import Reranker


# ── Prompt template ──────────────────────────────────────────────────────────

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_context_block(results: List[RetrievalResult]) -> str:
    """Format retrieved chunks into a numbered context block."""
    blocks = []
    for i, r in enumerate(results, 1):
        header = f"[Excerpt {i} | score={r.score:.3f}]"
        blocks.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def _build_user_message(query: str, context: str) -> str:
    return (
        f"Context excerpts from the document:\n\n{context}"
        f"\n\n---\n\nUser question: {query}"
    )


def _retrieve(vector_store, embedder, query, top_k, use_mmr, use_rerank=True):
    query_emb = embedder.embed_query(query)
    # ADDED: fetch a wider candidate pool when reranking so the cross-encoder
    # has something meaningful to re-sort, then trim to top_k after rerank.
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


# ── Groq client (process-level cache, framework-agnostic) ────────────────────

@functools.lru_cache(maxsize=8)
def _get_groq_client(api_key: str):
    """Return a cached Groq client. Cached per unique API key."""
    from groq import Groq
    return Groq(api_key=api_key)

@functools.lru_cache(maxsize=1)
def _get_reranker():
    return Reranker()

# ── Public RAG function (single-turn, no history) ────────────────────────────

def answer_query(
    query: str,
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
    Run the full RAG pipeline for a single question (no conversation history).

    Returns
    -------
    answer  : streaming generator (if stream=True) or full string (if stream=False)
    sources : list of RetrievalResult used as context
    """
    results = _retrieve(vector_store, embedder, query, top_k, use_mmr, use_rerank=True)

    if not results:
        no_doc = "No document has been indexed yet. Please upload a document first."
        return ((lambda: (yield no_doc))() if stream else no_doc), []

    context = _build_context_block(results)
    user_msg = _build_user_message(query, context)
    client = _get_groq_client(groq_api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

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


# ── Conversation-aware variant ───────────────────────────────────────────────
# FIX: added `stream` parameter so this matches answer_query()'s interface.
# backend/state.py and backend/api/routes.py call this with stream=True/False —
# without this parameter, FastAPI's POST /ask (stream=False) crashed with
# "TypeError: unexpected keyword argument 'stream'".

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
    RAG answer that includes prior conversation turns so the model can
    handle follow-up questions correctly.

    stream=True  → returns a token generator (for Streamlit UI / SSE)
    stream=False → returns the full answer string (for a plain JSON API response)

    Works in Streamlit, FastAPI (sync /ask or SSE /ask/stream), or CLI.
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
    for turn in history[-6:]:   # last 6 turns to stay within context window
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