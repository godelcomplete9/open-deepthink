"""Ingest: arbitrary formal text from .txt/.md/.tex or PDF.

Formal artifacts that follow formal structures — code repositories, math
models, physics models, documentation — enter the RSI loop here.
PDF extraction prefers PyMuPDF (already an open-deepthink dependency) and
falls back to pypdf when available.
"""

from __future__ import annotations

import os


def load_formal_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".pdf"):
        text = _extract_pdf_pymupdf(path)
        if text is None:
            text = _extract_pdf_pypdf(path)
        if text is None:
            raise RuntimeError(
                "PDF extraction requires PyMuPDF (fitz) or pypdf. "
                "Install one of them to ingest PDF artifacts."
            )
        return text
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_pdf_pymupdf(path: str) -> str | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    try:
        doc = fitz.open(path)
        pages = [(page.get_text() or "") for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except Exception:
        return None


def _extract_pdf_pypdf(path: str) -> str | None:
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    try:
        reader = PdfReader(path)
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return None
