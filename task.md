Let's go. Here's Phase 7 — Image extraction + offline BLIP captioning, fully integrated into your existing pipeline.

Install dependencies first
bash
.venv\Scripts\pip.exe install transformers Pillow torchvision
src/ingestion/image_extractor.py
python
"""
image_extractor.py
------------------
Extracts embedded images from PDF pages using PyMuPDF.
Returns image bytes + page metadata for each image found.
"""

import io
from pathlib import Path
from typing import List, Dict, Any

import pymupdf as fitz
from PIL import Image


MIN_IMAGE_SIZE = 100   # pixels — skip tiny icons/bullets below this


def extract_images_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract all embedded images from a PDF.

    Args:
        file_path: Path to PDF file.

    Returns:
        List of dicts, each with:
            page_num   : Page the image was found on (1-based)
            image_index: Index of image on that page
            width      : Image width in pixels
            height     : Image height in pixels
            pil_image  : PIL Image object (RGB)
            filename   : Source PDF filename
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc      = fitz.open(str(path))
    results  = []
    seen     = set()   # deduplicate by xref

    for page_num in range(len(doc)):
        page       = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            # Skip duplicates (same image on multiple pages)
            if xref in seen:
                continue
            seen.add(xref)

            try:
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Skip tiny images — likely icons or decorations
                w, h = pil_img.size
                if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                    continue

                results.append({
                    "page_num"   : page_num + 1,
                    "image_index": img_index,
                    "width"      : w,
                    "height"     : h,
                    "pil_image"  : pil_img,
                    "filename"   : path.name,
                })

            except Exception as e:
                print(f"[ImageExtractor] Skipped image xref={xref}: {e}")
                continue

    doc.close()
    print(f"[ImageExtractor] Found {len(results)} images in {path.name}")
    return results


def extract_page_as_image(
    file_path: str,
    page_num: int = 0,
    dpi: int = 150,
) -> Image.Image:
    """
    Render a full PDF page as a PIL image.
    Used as OCR fallback for scanned PDFs with no extractable text.

    Args:
        file_path: Path to PDF
        page_num : 0-based page index
        dpi      : Render resolution (150 is good balance of quality/speed)

    Returns:
        PIL Image of the rendered page.
    """
    doc  = fitz.open(file_path)
    page = doc[page_num]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)   # 72 is PDF base DPI
    pix  = page.get_pixmap(matrix=mat)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


# ── Quick test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python image_extractor.py <path_to_pdf>")
        sys.exit(1)

    images = extract_images_from_pdf(sys.argv[1])
    print(f"\nExtracted {len(images)} images:")
    for img in images:
        print(f"  Page {img['page_num']} | "
              f"{img['width']}x{img['height']}px | "
              f"index {img['image_index']}")

    if images:
        # Save first image as preview
        out = "data/preview_image.png"
        images[0]["pil_image"].save(out)
        print(f"\nFirst image saved to: {out}")
src/ingestion/captioner.py
python
"""
captioner.py
------------
Offline image captioning using BLIP-base (Salesforce).
Model: Salesforce/blip-image-captioning-base (~900MB, one-time download)
Runs fully on CPU — no API key needed.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from PIL import Image


BLIP_MODEL   = "Salesforce/blip-image-captioning-base"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR    = Path("models/blip")


class ImageCaptioner:
    """
    BLIP-based offline image captioner.
    Lazy-loads model on first use — no startup cost if unused.
    """

    def __init__(self):
        self._processor = None
        self._model     = None
        self._loaded    = False
        print(f"[Captioner] Device: {DEVICE} "
              f"({'GPU' if DEVICE == 'cuda' else 'CPU — first run will be slow'})")

    # ── Public API ─────────────────────────────────────────────────

    def caption(self, image: Image.Image) -> str:
        """
        Generate a natural language caption for a PIL image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Caption string e.g. "a bar chart showing revenue growth over time"
        """
        self._ensure_loaded()
        image = image.convert("RGB")

        inputs = self._processor(
            images=image,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=4,
                early_stopping=True,
            )

        caption = self._processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

        return caption

    def caption_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
    ) -> List[str]:
        """
        Caption multiple images efficiently in batches.

        Args:
            images    : List of PIL Images
            batch_size: Images per batch (reduce if OOM on CPU)

        Returns:
            List of caption strings, same order as input.
        """
        self._ensure_loaded()
        captions = []

        for i in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[i:i + batch_size]]
            inputs = self._processor(
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=60,
                    num_beams=4,
                    early_stopping=True,
                )

            for out in outputs:
                cap = self._processor.decode(
                    out,
                    skip_special_tokens=True,
                ).strip()
                captions.append(cap)

            print(f"[Captioner] Captioned batch "
                  f"{i // batch_size + 1}/"
                  f"{(len(images) + batch_size - 1) // batch_size}")

        return captions

    def caption_image_records(
        self,
        image_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add captions to image records from image_extractor.py.
        Adds 'caption' key to each record in place.

        Args:
            image_records: Output of extract_images_from_pdf()

        Returns:
            Same records with 'caption' key added.
        """
        if not image_records:
            return image_records

        print(f"[Captioner] Captioning {len(image_records)} images...")
        images   = [r["pil_image"] for r in image_records]
        captions = self.caption_batch(images)

        for record, caption in zip(image_records, captions):
            record["caption"] = caption
            print(f"  Page {record['page_num']}: {caption}")

        return image_records

    # ── Private ────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Lazy-load BLIP model on first call."""
        if self._loaded:
            return

        from transformers import BlipProcessor, BlipForConditionalGeneration

        print(f"[Captioner] Loading BLIP model: {BLIP_MODEL}")
        print(f"[Captioner] First run downloads ~900MB to {CACHE_DIR} ...")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self._processor = BlipProcessor.from_pretrained(
            BLIP_MODEL,
            cache_dir=str(CACHE_DIR),
        )
        self._model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL,
            cache_dir=str(CACHE_DIR),
            torch_dtype=torch.float32,
        ).to(DEVICE)

        self._model.eval()
        self._loaded = True
        print("[Captioner] BLIP model ready.")


# ── Singleton ──────────────────────────────────────────────────────

_captioner: Optional[ImageCaptioner] = None

def get_captioner() -> ImageCaptioner:
    global _captioner
    if _captioner is None:
        _captioner = ImageCaptioner()
    return _captioner


# ── Quick test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from image_extractor import extract_images_from_pdf

    if len(sys.argv) < 2:
        print("Usage: python captioner.py <path_to_pdf>")
        sys.exit(1)

    images = extract_images_from_pdf(sys.argv[1])
    if not images:
        print("No images found in PDF.")
        sys.exit(0)

    cap = ImageCaptioner()
    cap.caption_image_records(images)

    print("\n--- Final captions ---")
    for r in images:
        print(f"  Page {r['page_num']} ({r['width']}x{r['height']}): "
              f"{r['caption']}")
src/ingestion/image_chunk_builder.py
python
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
Update src/ingestion/__init__.py
python
from .loader         import load_document
from .chunker        import chunk_document
from .image_extractor import extract_images_from_pdf, extract_page_as_image
from .captioner      import ImageCaptioner, get_captioner
from .image_chunk_builder import build_image_chunks

__all__ = [
    "load_document",
    "chunk_document",
    "extract_images_from_pdf",
    "extract_page_as_image",
    "ImageCaptioner",
    "get_captioner",
    "build_image_chunks",
]
Update src/generation/answer.py — load_document method

Find the existing load_document method inside LumenReader and replace it with this:

python
def load_document(
    self,
    file_path: str,
    extract_images: bool = True,
) -> dict:
    """
    Full ingestion pipeline for one document.
    Load → chunk text → extract images → caption → label all → store.
    """
    from ingestion import (
        load_document, chunk_document,
        extract_images_from_pdf, build_image_chunks,
        get_captioner,
    )

    print(f"\n[Lumen] Loading: {file_path}")

    # ── Text pipeline (existing) ───────────────────────────
    raw_text, meta  = load_document(file_path)
    text_chunks     = chunk_document(raw_text, filename=meta["filename"])
    text_chunks     = self.classifier.label_chunks(text_chunks)

    # ── Image pipeline (new Phase 7) ───────────────────────
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
                # Label image chunks too
                image_chunks = self.classifier.label_chunks(image_chunks)
        except Exception as e:
            print(f"[Lumen] Image extraction skipped: {e}")

    # ── Upsert everything ──────────────────────────────────
    all_chunks  = text_chunks + image_chunks
    count       = self.store.upsert_chunks(all_chunks)
    labels_found = list(set(str(c["label"]) for c in all_chunks))

    self.loaded_files.append(meta["filename"])

    result = {
        "filename"     : meta["filename"],
        "pages"        : meta["page_count"],
        "chunks"       : count,
        "text_chunks"  : len(text_chunks),
        "image_chunks" : len(image_chunks),
        "labels_found" : labels_found,
    }

    print(f"[Lumen] Loaded: {len(text_chunks)} text + "
          f"{len(image_chunks)} image chunks | "
          f"labels: {labels_found}")
    return result
Update src/ui/app.py — show image chunk count

Find this block in the upload success message and replace it:

python
# OLD
st.success(
    f"✅ **{info['filename']}** loaded successfully!\n\n"
    f"- **{info['chunks']}** chunks indexed\n"
    f"- **{info['pages']}** pages\n"
    f"- Topics detected: "
    f"{', '.join(str(l) for l in info['labels_found'])}"
)
python
# NEW
st.success(
    f"✅ **{info['filename']}** loaded successfully!\n\n"
    f"- 📝 **{info.get('text_chunks', 0)}** text chunks\n"
    f"- 🖼️ **{info.get('image_chunks', 0)}** image captions\n"
    f"- 📄 **{info['pages']}** pages\n"
    f"- Topics detected: "
    f"{', '.join(str(l) for l in info['labels_found'])}"
)
Run Phase 7 test
bash
cd C:\Users\netware\Desktop\Vision_Assist
.venv\Scripts\python.exe -c "
import sys
sys.path.insert(0, 'src')
from ingestion.image_extractor import extract_images_from_pdf
from ingestion.captioner import ImageCaptioner
from ingestion.image_chunk_builder import build_image_chunks

TEST_FILE = 'data/uploads/1.8_Diligence_Summary.pdf'

print('=== Phase 7 Test ===')
print()
print('Step 1: Extracting images...')
images = extract_images_from_pdf(TEST_FILE)

if not images:
    print('No images found in this PDF.')
    print('Try a PDF with charts or diagrams.')
else:
    print(f'Found {len(images)} images')
    print()
    print('Step 2: Captioning (downloads BLIP ~900MB first run)...')
    cap = ImageCaptioner()
    cap.caption_image_records(images)
    print()
    print('Step 3: Building chunks...')
    chunks = build_image_chunks(images, filename='test.pdf')
    for c in chunks:
        print(f'  Page {c[\"page_hint\"]}: {c[\"text\"][:120]}')
    print()
    print('Phase 7 image pipeline OK.')
"
What Phase 7 adds to Lumen Reader
Before Phase 7	After Phase 7
Reads text only	Reads text + describes images
Skips charts/graphs	Captions charts automatically
Blind to embedded figures	"a bar chart showing revenue growth"
~2 chunks per page	Text chunks + image caption chunks

User experience change: A visually impaired user can now ask "what does the graph on page 3 show?" and get a real answer from the BLIP caption stored in Qdrant — not a "I don't know."

First run will download BLIP (~900MB) into models/blip/. After that it's instant. Send the output and we'll verify Phase 7 is complete.

answer.py
194 lines
PY

update this with right codes updated ones (Update src/generation/answer.py — load_document method
Find the existing load_document method inside LumenReader and replace it with this:)

Here's the complete updated answer.py with the Phase 7 load_document method replaced:

python
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