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

        if not image_list:
            # Fallback: check if the page contains vector graphics/drawings
            drawings = page.get_drawings()
            if len(drawings) > 0:
                try:
                    mat  = fitz.Matrix(150 / 72, 150 / 72)
                    pix  = page.get_pixmap(matrix=mat)
                    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    results.append({
                        "page_num"   : page_num + 1,
                        "image_index": 0,
                        "width"      : pix.width,
                        "height"     : pix.height,
                        "pil_image"  : pil_img,
                        "filename"   : path.name,
                    })
                    print(f"[ImageExtractor] Captured page {page_num + 1} as image (Vector Graphics detected)")
                except Exception as e:
                    print(f"[ImageExtractor] Skipped vector page {page_num + 1}: {e}")
            continue

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