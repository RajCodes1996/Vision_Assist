"""
loader.py
---------
Document loader for Lumen Reader.
Supports: PDF, DOCX, TXT, Markdown, HTML
Returns: raw text string + document metadata dict
"""

import os
import pymupdf as fitz
from pathlib import Path
from typing import Tuple


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html"}


def load_document(file_path: str) -> Tuple[str, dict]:
    """
    Load a document from disk and return (raw_text, metadata).

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        raw_text : Full extracted text, cleaned and normalised.
        metadata : Dict with keys — filename, extension, page_count, file_size_kb.

    Raises:
        ValueError: If file extension is not supported.
        FileNotFoundError: If file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    metadata = {
        "filename": path.name,
        "extension": ext,
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "page_count": 0,
    }

    if ext == ".pdf":
        raw_text, page_count = _load_pdf(str(path))
        metadata["page_count"] = page_count

    elif ext == ".docx":
        raw_text = _load_docx(str(path))

    elif ext in {".txt", ".md"}:
        raw_text = _load_plaintext(str(path))

    elif ext == ".html":
        raw_text = _load_html(str(path))

    raw_text = _clean_text(raw_text)
    return raw_text, metadata


# ── Private loaders ────────────────────────────────────────────────

def _load_pdf(file_path: str) -> Tuple[str, int]:
    doc = fitz.open(file_path)
    pages = []
    page_count = len(doc)                # ✅ read BEFORE close
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()
    return "\n\n".join(pages), page_count

def _load_docx(file_path: str) -> str:
    """Extract text from DOCX. Falls back gracefully if python-docx missing."""
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX support. "
            "Install it: pip install python-docx"
        )


def _load_plaintext(file_path: str) -> str:
    """Read plain text / markdown files."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_html(file_path: str) -> str:
    """Strip HTML tags and return plain text."""
    try:
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n")
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is required for HTML support. "
            "Install it: pip install beautifulsoup4"
        )


def _clean_text(text: str) -> str:
    """
    Normalise extracted text:
    - Collapse multiple blank lines to a single blank line
    - Strip leading/trailing whitespace per line
    - Remove null bytes
    """
    import re
    text = text.replace("\x00", "")                    # remove null bytes
    text = re.sub(r"\r\n", "\n", text)                 # normalise line endings
    text = re.sub(r"[ \t]+", " ", text)                # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)             # max 2 consecutive newlines
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ── Quick test (run directly) ──────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python loader.py <path_to_document>")
        sys.exit(1)

    text, meta = load_document(sys.argv[1])
    print(f"\n--- Metadata ---")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"\n--- Text preview (first 500 chars) ---")
    print(text[:500])
    print(f"\n--- Total characters: {len(text)} ---")