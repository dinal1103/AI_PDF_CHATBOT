"""
Document Processor
──────────────────
Handles text extraction from PDF / DOCX / TXT files and splits
the extracted text into overlapping chunks ready for embedding.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
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
    name: str
    file_type: str
    total_pages: int = 0
    total_chars: int = 0
    total_words: int = 0
    total_chunks: int = 0
    language: str = "en"
    extra: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# FIXED: now returns per-page text list instead of one joined string,
# so chunks can carry an accurate page number instead of always None/wrong.
def extract_from_pdf(file_bytes: bytes) -> tuple[list[str], int]:
    """Return (list_of_cleaned_page_texts, page_count) from PDF bytes."""
    try:
        import pdfplumber
        pages_text: list[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text() or ""
                pages_text.append(_clean_text(t))
        return pages_text, n_pages
    except Exception:
        pass

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = [_clean_text(page.get_text()) for page in doc]
        return pages_text, len(doc)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}") from e


def extract_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _clean_text("\n\n".join(paragraphs))
    except Exception as e:
        raise RuntimeError(f"DOCX extraction failed: {e}") from e


def extract_from_txt(file_bytes: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return _clean_text(file_bytes.decode(enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode text file.")


class SemanticChunker:
    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _sentences(self, text: str) -> list[str]:
        parts = self._SENTENCE_END.split(text)
        return [p.strip() for p in parts if p.strip()]

    # FIXED: split() now accepts an optional `page` so each chunk from a
    # single page gets tagged correctly, instead of chunking the whole
    # joined document and losing page boundaries.
    def split(self, text: str, source: str = "", page: Optional[int] = None,
              start_chunk_id: int = 0) -> list[Chunk]:
        sentences = self._sentences(text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        chunk_id = start_chunk_id

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(Chunk(text=chunk_text, chunk_id=chunk_id, source=source, page=page))
                chunk_id += 1

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
            chunks.append(Chunk(text=" ".join(current), chunk_id=chunk_id, source=source, page=page))

        return chunks


def process_document(
    uploaded_file,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> tuple[list[Chunk], DocumentInfo]:
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower()
    file_bytes = uploaded_file.read()

    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pages = 0
    all_chunks: list[Chunk] = []

    if ext == "pdf":
        # FIXED: chunk per-page so page numbers are accurate, then
        # continue chunk_id sequentially across pages.
        pages_text, pages = extract_from_pdf(file_bytes)
        full_text_for_stats = "\n\n".join(pages_text)
        next_id = 0
        for page_num, page_text in enumerate(pages_text, start=1):
            if not page_text.strip():
                continue
            page_chunks = chunker.split(page_text, source=name, page=page_num, start_chunk_id=next_id)
            all_chunks.extend(page_chunks)
            next_id += len(page_chunks)
        text = full_text_for_stats

    elif ext in ("docx", "doc"):
        text = extract_from_docx(file_bytes)
        all_chunks = chunker.split(text, source=name, page=None)

    elif ext == "txt":
        text = extract_from_txt(file_bytes)
        all_chunks = chunker.split(text, source=name, page=None)

    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    if not text.strip():
        raise ValueError("No text could be extracted from the document.")

    doc_info = DocumentInfo(
        name=name,
        file_type=ext.upper(),
        total_pages=pages,
        total_chars=len(text),
        total_words=len(text.split()),
        total_chunks=len(all_chunks),
    )

    return all_chunks, doc_info