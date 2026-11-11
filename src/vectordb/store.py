"""
store.py
--------
Qdrant vector store for Lumen Reader.
Handles: collection setup, chunk upsert, hybrid search, metadata filter.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from qdrant_client import models
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "lumen_reader")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
VECTOR_SIZE       = 768  # BGE-base output dimension


class VectorStore:
    """
    Qdrant-backed vector store with label-aware metadata filtering.
    """

    def __init__(self, in_memory: bool = False):
        """
        Args:
            in_memory: If True, uses Qdrant in-memory mode (no Docker needed).
                       Set to False for production with Docker running.
        """
        if in_memory:
            self.client = QdrantClient(":memory:")
            print("[VectorStore] Running in-memory mode (no Docker)")
        else:
            self.client = QdrantClient(url=QDRANT_URL, timeout=10)
            print(f"[VectorStore] Connected to Qdrant at {QDRANT_URL}")

        print(f"[VectorStore] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._ensure_collection()

    # ── Collection setup ───────────────────────────────────────────

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[VectorStore] Created collection: {COLLECTION_NAME}")
        else:
            print(f"[VectorStore] Collection exists: {COLLECTION_NAME}")

    # ── Upsert ─────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and upsert a list of labelled chunks into Qdrant.
        Chunks must have 'label' key — run labeller.label_chunks() first.

        Args:
            chunks: List of chunk dicts (from chunker + labeller)

        Returns:
            Number of points upserted.
        """
        if not chunks:
            print("[VectorStore] No chunks to upsert.")
            return 0

        texts = [c["text"] for c in chunks]
        print(f"[VectorStore] Embedding {len(texts)} chunks...")
        vectors = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity needs normalised vecs
        )

        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=self._chunk_id_to_uuid(chunk["chunk_id"]),
                vector=vector.tolist(),
                payload={
                    "chunk_id"        : chunk["chunk_id"],
                    "text"            : chunk["text"],
                    "filename"        : chunk["filename"],
                    "chunk_index"     : chunk["chunk_index"],
                    "page_hint"       : chunk["page_hint"],
                    "word_count"      : chunk["word_count"],
                    "label"           : chunk.get("label", "General"),
                    "label_confidence": chunk.get("label_confidence", 0.0),
                },
            ))

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[VectorStore] Upserted {len(points)} points.")
        return len(points)

    # ── Search ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        label_filter: Optional[str] = None,
        filename_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional label + filename filtering.

        Args:
            query          : User question string
            top_k          : Number of results to return
            label_filter   : Only return chunks with this label (e.g. "Finance")
            filename_filter: Only return chunks from this document

        Returns:
            List of result dicts with text, score, and metadata.
        """
        query_vector = self.embedder.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # Build filter conditions
        conditions = []
        if label_filter:
            conditions.append(FieldCondition(
                key="label",
                match=MatchValue(value=label_filter),
            ))
        if filename_filter:
            conditions.append(FieldCondition(
                key="filename",
                match=MatchValue(value=filename_filter),
            ))

        search_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector, 
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        # FIX: query_points returns a structural object containing a points list
        return [
            {
                "score"           : round(r.score, 4),
                "text"            : r.payload["text"],
                "chunk_id"        : r.payload["chunk_id"],
                "filename"        : r.payload["filename"],
                "page_hint"       : r.payload["page_hint"],
                "label"           : r.payload["label"],
                "label_confidence": r.payload["label_confidence"],
                "chunk_index"     : r.payload["chunk_index"],
            }
            for r in results.points
        ]

    # ── Utility ────────────────────────────────────────────────────

    def get_collection_info(self) -> Dict[str, Any]:
        """Return stats about the current collection."""
        info = self.client.get_collection(COLLECTION_NAME)
        return {
            "name"         : COLLECTION_NAME,
            "total_points" : info.points_count,
            "vector_size"  : VECTOR_SIZE,
            "distance"     : "cosine",
        }

    def delete_collection(self):
        """Wipe the collection — use carefully."""
        self.client.delete_collection(COLLECTION_NAME)
        print(f"[VectorStore] Deleted collection: {COLLECTION_NAME}")
        self._ensure_collection()

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        """Convert 16-char hex chunk_id to a valid UUID for Qdrant."""
        padded = chunk_id.ljust(32, "0")
        return str(uuid.UUID(padded))


# ── Quick test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    from ingestion import load_document, chunk_document
    from labelling.labeller import LabelClassifier

    TEST_FILE = "data/uploads/1.8_Diligence_Summary.pdf"

    # 1. Load + chunk
    print("\n--- Step 1: Load & chunk ---")
    raw_text, meta = load_document(TEST_FILE)
    chunks = chunk_document(raw_text, filename=meta["filename"])
    print(f"  Chunks: {len(chunks)}")

    # 2. Label
    print("\n--- Step 2: Label chunks ---")
    clf = LabelClassifier()
    clf.train()
    clf.save()
    chunks = clf.label_chunks(chunks)
    for c in chunks:
        print(f"  [{c['label']:15s}] ({c['label_confidence']:.2f})  {c['text'][:60]}...")

    # 3. Upsert to Qdrant (in-memory for test)
    print("\n--- Step 3: Upsert to Qdrant ---")
    store = VectorStore(in_memory=True)
    store.upsert_chunks(chunks)
    print(store.get_collection_info())

    # 4. Search
    print("\n--- Step 4: Search ---")
    results = store.search("What is the financial summary?", top_k=3)
    for r in results:
        print(f"\n[{r['label']}] (Score: {r['score']})")
        print(f"Text: {r['text'][:150]}...")
