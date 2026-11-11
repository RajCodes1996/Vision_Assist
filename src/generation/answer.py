"""
answer.py
---------
Answer generation wrapper for Lumen Reader.
Ties together: RAG graph output + TTS playback + citation formatting.
This is the single entry point the UI calls.
"""

import sys
import os
from typing import Optional, List
from pathlib import Path

# Resolve project root and src, add both to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_dir      = os.path.join(project_root, "src")
os.chdir(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dotenv import load_dotenv
load_dotenv()


class LumenReader:
    """
    Main orchestrator for Lumen Reader.
    Wraps the full pipeline: ingest → label → store → retrieve → speak.
    """

    def __init__(self, in_memory: bool = True, tts_enabled: bool = True):
        """
        Args:
            in_memory  : Use Qdrant in-memory (True) or Docker (False)
            tts_enabled: Auto-speak answers aloud
        """
        from labelling.labeller import LabelClassifier
        from vectordb.store import VectorStore
        from retrieval.graph import build_rag_graph
        from accessibility.tts import TTSEngine

        print("[Lumen] Initialising pipeline...")

        # Core components
        self.classifier  = LabelClassifier()
        self.store       = VectorStore(in_memory=in_memory)
        self.tts         = TTSEngine() if tts_enabled else None
        self.app         = build_rag_graph(self.store, self.classifier)

        # State
        self.history         : List[dict] = []
        self.loaded_files    : List[str]  = []
        self.tts_enabled                  = tts_enabled

        # Train classifier on seed data
        model_path = Path("models/label_classifier.pkl")
        if model_path.exists():
            self.classifier.load()
        else:
            self.classifier.train()
            self.classifier.save()

        print("[Lumen] Ready.")

    # ── Document ingestion ─────────────────────────────────────────

    def load_document(
        self,
        file_path: str,
        extract_images: bool = True,
    ) -> dict:
        """
        Full ingestion pipeline for one document.
        Load → chunk text → extract images → caption → label all → store.

        Args:
            file_path      : Path to PDF/DOCX/TXT/MD/HTML
            extract_images : Run BLIP image captioning (PDF only)

        Returns:
            Dict with filename, pages, chunks, text_chunks,
            image_chunks, labels_found.
        """
        from ingestion import (
            load_document, chunk_document,
            extract_images_from_pdf, build_image_chunks,
            get_captioner,
        )

        print(f"\n[Lumen] Loading: {file_path}")

        # ── Text pipeline (Phases 2-3) ─────────────────────────────
        raw_text, meta = load_document(file_path)
        text_chunks    = chunk_document(raw_text, filename=meta["filename"])
        text_chunks    = self.classifier.label_chunks(text_chunks)

        # ── Image pipeline (Phase 7) ───────────────────────────────
        image_chunks = []
        if extract_images and file_path.lower().endswith(".pdf"):
            try:
                image_records = extract_images_from_pdf(file_path)
                if image_records:
                    captioner     = get_captioner()
                    image_records = captioner.caption_image_records(image_records)
                    image_chunks  = build_image_chunks(
                        image_records,
                        filename    =meta["filename"],
                        start_index =len(text_chunks),
                    )
                    image_chunks = self.classifier.label_chunks(image_chunks)
                else:
                    print("[Lumen] No images found in PDF — text only.")
            except Exception as e:
                print(f"[Lumen] Image extraction skipped: {e}")

        # ── Upsert everything into Qdrant ──────────────────────────
        all_chunks   = text_chunks + image_chunks
        count        = self.store.upsert_chunks(all_chunks)
        labels_found = list(set(str(c["label"]) for c in all_chunks))

        self.loaded_files.append(meta["filename"])

        result = {
            "filename"    : meta["filename"],
            "pages"       : meta["page_count"],
            "chunks"      : count,
            "text_chunks" : len(text_chunks),
            "image_chunks": len(image_chunks),
            "labels_found": labels_found,
        }

        print(f"[Lumen] Loaded: {len(text_chunks)} text + "
              f"{len(image_chunks)} image chunks | "
              f"labels: {labels_found}")
        return result

    # ── Query ──────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        filename_filter: Optional[str] = None,
        speak: bool = True,
    ) -> dict:
        """
        Ask a question and get an answer with citations.

        Args:
            question       : User's question
            filename_filter: Restrict to one document (optional)
            speak          : Read answer aloud via TTS

        Returns:
            Dict with answer, sources, detected_label, history.
        """
        from retrieval.graph import run_query

        print(f"\n[Lumen] Question: {question}")

        result = run_query(
            self.app,
            question=question,
            history=self.history,
            filename_filter=filename_filter,
        )

        # Update conversation memory
        self.history = result["history"]

        # Speak the answer
        if speak and self.tts_enabled and self.tts:
            self.tts.speak_answer(result["answer"], result["sources"])

        return result

    def format_sources(self, sources: list) -> str:
        """Format source citations as readable text."""
        if not sources:
            return "No sources found."
        lines = []
        for i, s in enumerate(sources, 1):
            lines.append(
                f"[{i}] {s['filename']} | "
                f"Page {s['page']} | "
                f"Topic: {s['label']} | "
                f"Score: {s['score']}"
            )
        return "\n".join(lines)

    def reset_memory(self):
        """Clear conversation history."""
        self.history = []
        print("[Lumen] Conversation memory cleared.")

    def reset_store(self):
        """Wipe vector store and reload."""
        self.store.delete_collection()
        self.loaded_files = []
        print("[Lumen] Vector store reset.")


# ── Quick test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_FILE = "data/uploads/1.8_Diligence_Summary.pdf"

    print("=== Lumen Reader — Full Pipeline Test ===\n")
    reader = LumenReader(in_memory=True, tts_enabled=True)

    # Load document (now includes image captioning)
    info = reader.load_document(TEST_FILE)
    print(f"\nDocument loaded:")
    print(f"  File         : {info['filename']}")
    print(f"  Pages        : {info['pages']}")
    print(f"  Text chunks  : {info['text_chunks']}")
    print(f"  Image chunks : {info['image_chunks']}")
    print(f"  Total chunks : {info['chunks']}")
    print(f"  Labels       : {info['labels_found']}")

    # Ask questions including one about images
    questions = [
        "What is AI diligence?",
        "What are the 4 Ds mentioned in the document?",
        "What do the images or figures in the document show?",
        "Can you summarise what we discussed?",
    ]

    for q in questions:
        print(f"\n{'─'*50}")
        print(f"Q: {q}")
        result = reader.ask(q, speak=True)
        print(f"A: {result['answer'][:250]}...")
        print(f"\nSources:\n{reader.format_sources(result['sources'])}")

    print("\n=== Phase 7 complete ===")