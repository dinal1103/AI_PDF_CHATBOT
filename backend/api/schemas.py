"""
backend/api/schemas.py
──────────────────────
Pydantic models for all FastAPI request bodies and responses.

Pydantic validates incoming JSON automatically. If a required field
is missing or wrong type, FastAPI returns a 422 error with details.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Upload response ──────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    status: str
    filename: str
    file_type: str
    total_chunks: int
    total_words: int
    total_pages: int
    message: str


# ── Ask request / response ───────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    model: str = Field(default="llama-3.3-70b-versatile")
    top_k: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    use_mmr: bool = Field(default=True)
    history: List[dict] = Field(default_factory=list)


class SourceChunk(BaseModel):
    chunk_id: int
    text: str
    score: float
    source: str
    page: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    model: str
    chunks_retrieved: int


# ── Status ───────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str                    # "ready" | "no_document"
    doc_name: Optional[str]
    total_chunks: int
    embed_model: str