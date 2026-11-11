"""
image_chunk_builder.py
----------------------
Converts captioned image records into chunk dicts
compatible with the existing Qdrant pipeline from Phase 3.
These chunks flow directly into store.upsert_chunks().
"""

import hashlib
from typing import List, Dict, Any


def build_image_chunks(
    image_records: List[Dict[str, Any]],
    filename: str,
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    """
    Convert captioned image records into chunk dicts.

    Args:
        image_records: Output of captioner.caption_image_records()
                       Each must have 'caption' key.
        filename     : Source document filename
        start_index  : Chunk index offset (to avoid collision
                       with text chunk indices from chunker.py)

    Returns:
        List of chunk dicts ready for labeller + store.upsert_chunks()
    """
    chunks = []

    for i, record in enumerate(image_records):
        caption = record.get("caption", "").strip()
        if not caption:
            continue

        # Build rich text for embedding — context helps retrieval
        text = (
            f"[Image on Page {record['page_num']}] "
            f"{caption}. "
            f"This image appears on page {record['page_num']} "
            f"of the document {filename}."
        )

        chunk_id = _make_image_chunk_id(
            filename,
            record["page_num"],
            record["image_index"],
        )

        chunks.append({
            "chunk_id"   : chunk_id,
            "text"       : text,
            "filename"   : filename,
            "chunk_index": start_index + i,
            "page_hint"  : record["page_num"],
            "word_count" : len(text.split()),
            "chunk_type" : "image_caption",    # extra metadata
            "img_width"  : record["width"],
            "img_height" : record["height"],
        })

    print(f"[ImageChunkBuilder] Built {len(chunks)} image caption chunks")
    return chunks


def _make_image_chunk_id(
    filename: str,
    page_num: int,
    img_index: int,
) -> str:
    raw = f"{filename}::img::page{page_num}::idx{img_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]