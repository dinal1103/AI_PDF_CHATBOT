"""
backend/api/routes.py
─────────────────────
All FastAPI route handlers.

Routes
------
POST /upload        → ingest a document (PDF/DOCX/TXT)
POST /ask           → non-streaming Q&A
GET  /ask/stream    → streaming Q&A via Server-Sent Events
GET  /status        → check if a document is loaded
DELETE /document    → clear the current document + index
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .schemas import AskRequest, AskResponse, SourceChunk, StatusResponse, UploadResponse
from ..state import AppState

router = APIRouter()

# Singleton app state (vector store + embedder live here)
_state = AppState()


# ── POST /upload ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    Accept a file upload, extract text, embed chunks, build FAISS index.
    Replaces any previously loaded document.
    """
    allowed = {"pdf", "docx", "doc", "txt"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Allowed: {allowed}",
        )

    # Read bytes — UploadFile is async
    file_bytes = await file.read()

    # Wrap bytes in a file-like object that process_document expects
    import io

    class _FakeSLFile:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self._data = data

        def read(self) -> bytes:
            return self._data

    fake_file = _FakeSLFile(file.filename, file_bytes)

    try:
        doc_info = _state.ingest(fake_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return UploadResponse(
        status="ok",
        filename=doc_info.name,
        file_type=doc_info.file_type,
        total_chunks=doc_info.total_chunks,
        total_words=doc_info.total_words,
        total_pages=doc_info.total_pages,
        message=f"Document indexed. {doc_info.total_chunks} chunks ready.",
    )


# ── POST /ask ────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    """Non-streaming Q&A. Returns full answer once LLM finishes."""
    if not _state.is_ready():
        raise HTTPException(status_code=400, detail="No document loaded. POST /upload first.")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set on server.")

    try:
        answer, sources = _state.answer(
            query=payload.query,
            history=payload.history,
            groq_api_key=api_key,
            model=payload.model,
            top_k=payload.top_k,
            temperature=payload.temperature,
            use_mmr=payload.use_mmr,
            stream=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        answer=answer,
        sources=[
            SourceChunk(
                chunk_id=s.chunk_id,
                text=s.text,
                score=round(s.score, 4),
                source=s.source,
                page=s.page,
            )
            for s in sources
        ],
        model=payload.model,
        chunks_retrieved=len(sources),
    )


# ── GET /ask/stream ──────────────────────────────────────────────────────────

@router.get("/ask/stream")
async def ask_stream(
    query: str,
    model: str = "llama3-8b-8192",
    top_k: int = 5,
    temperature: float = 0.2,
    use_mmr: bool = True,
):
    """
    Streaming Q&A via Server-Sent Events (SSE).
    Client reads the event stream and appends tokens.

    Usage from JS:
        const es = new EventSource('/ask/stream?query=...')
        es.onmessage = (e) => appendToken(e.data)
    """
    if not _state.is_ready():
        raise HTTPException(status_code=400, detail="No document loaded.")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set.")

    token_gen, _ = _state.answer(
        query=query,
        history=[],
        groq_api_key=api_key,
        model=model,
        top_k=top_k,
        temperature=temperature,
        use_mmr=use_mmr,
        stream=True,
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        for token in token_gen:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── GET /status ───────────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
async def status():
    """Check if a document is loaded and ready for questions."""
    info = _state.get_info()
    return StatusResponse(
        status="ready" if _state.is_ready() else "no_document",
        doc_name=info.get("doc_name"),
        total_chunks=info.get("total_chunks", 0),
        embed_model=info.get("embed_model", ""),
    )


# ── DELETE /document ──────────────────────────────────────────────────────────

@router.delete("/document")
async def clear_document():
    """Clear the loaded document and reset the vector index."""
    _state.reset()
    return {"status": "cleared", "message": "Document and index removed."}