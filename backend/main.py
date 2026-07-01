"""
backend/main.py
───────────────
FastAPI application factory.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Interactive API docs:
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocuMind AI — Backend API",
        description="RAG-powered document Q&A. Upload a doc, ask questions.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow Streamlit frontend (port 8501) and any localhost dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",   # Streamlit default
            "http://localhost:3000",   # React dev server (future)
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/", tags=["health"])
    async def root():
        return {"status": "ok", "service": "DocuMind AI API", "version": "1.0.0"}

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "healthy"}

    return app


app = create_app()