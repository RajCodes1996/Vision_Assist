"""
chunker.py
----------
Semantic chunker for Lumen Reader.
Splits raw document text into overlapping, structure-aware chunks.
Each chunk carries: text, chunk_id, page_hint, start_char, end_char.
"""

import re
import hashlib
from typing import List, Dict, Any


# ── Config defaults (override by passing kwargs) ───────────────────

DEFAULT_CHUNK_SIZE    = 512   # target characters per chunk
DEFAULT_CHUNK_OVERLAP = 80    # overlap between consecutive chunks
DEFAULT_MIN_CHUNK     = 100   # discard chunks shorter than this


def chunk_document(
    raw_text: str,
    filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_len: int = DEFAULT_MIN_CHUNK,
) -> List[Dict[str, Any]]:
    """
    Split raw_text into semantic chunks.

    Args:
        raw_text     : Full document text from loader.py
        filename     : Source filename (used to build chunk IDs)
        chunk_size   : Target character length per chunk
        chunk_overlap: How many chars the next chunk reuses from the previous
        min_chunk_len: Chunks shorter than this are discarded

    Returns:
        List of chunk dicts, each with:
            chunk_id   : Deterministic SHA-256 based ID
            text       : Chunk content
            filename   : Source document name
            chunk_index: Sequential index (0-based)
            page_hint  : Best-guess page number (parsed from [Page N] markers)
            start_char : Start position in raw_text
            end_char   : End position in raw_text
            word_count : Number of words in chunk
    """
    # Split into paragraphs first (structure-aware)
    paragraphs = _split_into_paragraphs(raw_text)

    # Merge short paragraphs into chunks respecting chunk_size
    raw_chunks = _merge_paragraphs_into_chunks(
        paragraphs, chunk_size, chunk_overlap
    )

    # Build final chunk dicts
    chunks = []
    cursor = 0
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if len(chunk_text) < min_chunk_len:
            continue

        # Find approximate position in raw_text
        start = raw_text.find(chunk_text[:60], cursor)
        if start == -1:
            start = cursor
        end = start + len(chunk_text)
        cursor = max(cursor, end - chunk_overlap)

        chunks.append({
            "chunk_id"   : _make_chunk_id(filename, idx, chunk_text),
            "text"       : chunk_text,
            "filename"   : filename,
            "chunk_index": idx,
            "page_hint"  : _extract_page_hint(chunk_text),
            "start_char" : start,
            "end_char"   : end,
            "word_count" : len(chunk_text.split()),
        })

    return chunks


# ── Private helpers ────────────────────────────────────────────────


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text on double newlines (paragraph breaks).
    Also splits on detected [Page N] markers inserted by loader.py.
    """
    # Treat [Page N] markers as paragraph separators
    text = re.sub(r"\[Page \d+\]", lambda m: f"\n\n{m.group()}\n\n", text)
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _merge_paragraphs_into_chunks(
    paragraphs: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Greedily merge consecutive paragraphs until chunk_size is reached.
    When a paragraph is too long, sentence-split it and merge sentences.
    """
    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        # If single paragraph exceeds chunk_size, sentence-split it first
        if len(para) > chunk_size:
            # Flush what we have
            if current_parts:
                chunks.append(" ".join(current_parts))
                # Carry overlap
                overlap_text = " ".join(current_parts)[-chunk_overlap:]
                current_parts = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)

            # Sentence split the long paragraph
            sentences = _split_sentences(para)
            for sentence in sentences:
                if current_len + len(sentence) > chunk_size and current_parts:
                    chunks.append(" ".join(current_parts))
                    overlap_text = " ".join(current_parts)[-chunk_overlap:]
                    current_parts = [overlap_text] if overlap_text.strip() else []
                    current_len = len(overlap_text)
                current_parts.append(sentence)
                current_len += len(sentence) + 1
        else:
            if current_len + len(para) > chunk_size and current_parts:
                chunks.append(" ".join(current_parts))
                overlap_text = " ".join(current_parts)[-chunk_overlap:]
                current_parts = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)
            current_parts.append(para)
            current_len += len(para) + 1

    # Flush remaining
    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _split_sentences(text: str) -> List[str]:
    """
    Naive but robust sentence splitter.
    Splits on '. ', '! ', '? ' while avoiding common abbreviations.
    """
    # Protect common abbreviations
    text = re.sub(r"\b(Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|Fig|No)\.", r"\1<DOT>", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]
    return sentences


def _extract_page_hint(chunk_text: str) -> int:
    """
    Parse the [Page N] marker from chunk text to get a page hint.
    Returns 0 if no marker found.
    """
    match = re.search(r"\[Page (\d+)\]", chunk_text)
    return int(match.group(1)) if match else 0


def _make_chunk_id(filename: str, index: int, text: str) -> str:
    """
    Generate a deterministic chunk ID using SHA-256.
    Format: first 16 chars of hash — short but collision-safe for doc sizes.
    """
    raw = f"{filename}::{index}::{text[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Quick test (run directly) ──────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    from loader import load_document

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_document>")
        sys.exit(1)

    raw_text, meta = load_document(sys.argv[1])
    chunks = chunk_document(raw_text, filename=meta["filename"])

    print(f"\n--- Chunking complete ---")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Source file  : {meta['filename']}")

    print(f"\n--- First 3 chunks ---")
    for chunk in chunks[:3]:
        print(json.dumps({
            "chunk_id"   : chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "page_hint"  : chunk["page_hint"],
            "word_count" : chunk["word_count"],
            "preview"    : chunk["text"][:120] + "...",
        }, indent=2))