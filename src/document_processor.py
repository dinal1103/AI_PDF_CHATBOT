"""
Document Processor
──────────────────
Handles text extraction from PDF / DOCX / TXT files and splits
the extracted text into overlapping chunks ready for embedding.

Supported file types
---------------------
• PDF   → pdfplumber (layout-aware) with PyMuPDF fallback
• DOCX  → python-docx
• TXT   → plain read (UTF-8 → latin-1 → cp1252 fallback chain)

No framework dependency — works in Streamlit, FastAPI, or plain Python.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""
    text: str
    chunk_id: int
    page: Optional[int] = None
    source: str = ""
    word_count: int = 0
    char_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)


@dataclass
class DocumentInfo:
    """High-level document statistics."""
    name: str
    file_type: str
    total_pages: int = 0
    total_chars: int = 0
    total_words: int = 0
    total_chunks: int = 0
    language: str = "en"
    extra: dict = field(default_factory=dict)


# ── Text extraction helpers ──────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalise unicode, collapse whitespace, strip control chars."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    text = re.sub(r"\r\n|\r", "\n", text)                            # line endings
    text = re.sub(r"[ \t]+", " ", text)                              # horiz space
    text = re.sub(r"\n{3,}", "\n\n", text)                           # blank lines
    return text.strip()


def extract_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """Return (full_text, page_count) from PDF bytes."""
    try:
        import pdfplumber
        text_pages: list[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text() or ""
                text_pages.append(t)
        return _clean_text("\n\n".join(text_pages)), n_pages
    except Exception:
        pass

    # Fallback: PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_pages = [page.get_text() for page in doc]
        return _clean_text("\n\n".join(text_pages)), len(doc)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}") from e


def extract_from_docx(file_bytes: bytes) -> str:
    """Return full text from DOCX bytes."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _clean_text("\n\n".join(paragraphs))
    except Exception as e:
        raise RuntimeError(f"DOCX extraction failed: {e}") from e


def extract_from_txt(file_bytes: bytes) -> str:
    """Return full text from TXT bytes (tries UTF-8, then latin-1)."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return _clean_text(file_bytes.decode(enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode text file.")


# ── Chunker ──────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Splits text into overlapping chunks on sentence / paragraph boundaries.

    Strategy
    --------
    1. Split text into sentences (regex-based — no NLTK dependency).
    2. Greedily accumulate sentences until chunk_size chars is reached.
    3. Keep last N sentences as overlap for the next chunk.
    """

    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _sentences(self, text: str) -> list[str]:
        parts = self._SENTENCE_END.split(text)
        return [p.strip() for p in parts if p.strip()]

    def split(self, text: str, source: str = "") -> list[Chunk]:
        sentences = self._sentences(text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        chunk_id = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(Chunk(text=chunk_text, chunk_id=chunk_id, source=source))
                chunk_id += 1

                # Build overlap from tail sentences
                overlap_buf: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap_buf
                current_len = overlap_len

            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(Chunk(text=" ".join(current), chunk_id=chunk_id, source=source))

        return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def process_document(
    uploaded_file,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> tuple[list[Chunk], DocumentInfo]:
    """
    Extract text from an uploaded file object and split into chunks.
    Works with Streamlit UploadedFile or any file-like object with
    .name and .read() attributes.

    Returns
    -------
    chunks    : list[Chunk]
    doc_info  : DocumentInfo
    """
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower()
    file_bytes = uploaded_file.read()

    pages = 0
    if ext == "pdf":
        text, pages = extract_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        text = extract_from_docx(file_bytes)
    elif ext == "txt":
        text = extract_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    if not text.strip():
        raise ValueError("No text could be extracted from the document.")

    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split(text, source=name)

    doc_info = DocumentInfo(
        name=name,
        file_type=ext.upper(),
        total_pages=pages,
        total_chars=len(text),
        total_words=len(text.split()),
        total_chunks=len(chunks),
    )

    return chunks, doc_info