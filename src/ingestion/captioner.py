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