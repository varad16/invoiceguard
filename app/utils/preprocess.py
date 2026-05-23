"""
Document preprocessing.

Two responsibilities:
  1. Normalize input to a PIL Image (PDF → image via pdf2image, otherwise
     load directly with Pillow).
  2. Run OCR via pytesseract and return the raw text. This is the fallback
     when LayoutLMv3 isn't available, and it's also handy for the duplicate
     detector which uses fuzzy text matching.

Both pdf2image and pytesseract are heavy dependencies (poppler, tesseract).
The module guards them behind try-imports so tests can run with mocked PIL
objects on machines without the binaries installed.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Optional

from PIL import Image


def file_sha256(data: bytes) -> str:
    """Stable content hash — used by the duplicate detector for exact matches."""
    return hashlib.sha256(data).hexdigest()


def load_invoice_image(file_bytes: bytes, filename: str) -> Image.Image:
    """Convert an uploaded invoice (PDF or image) to a PIL Image.

    For multi-page PDFs we use page 1 only — invoices that span pages have
    their summary on page 1 in 99% of real-world cases. Multi-page handling
    is a future-work bullet, not a hackathon-scope feature.
    """
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pdf2image import convert_from_bytes  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "pdf2image (and poppler) required for PDF input. "
                "Install: brew install poppler && pip install pdf2image"
            ) from e
        pages = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)
        if not pages:
            raise ValueError("PDF contained no pages")
        return pages[0].convert("RGB")
    # Image input — Pillow handles JPEG, PNG, TIFF, etc.
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def run_ocr(image: Image.Image) -> str:
    """OCR via tesseract. Returns the full extracted text in reading order.

    Falls back to an empty string if tesseract isn't installed — downstream
    code is expected to degrade gracefully (LayoutLMv3 stub can still run,
    fuzzy duplicate matching just gets less material to work with).
    """
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(image)
    except Exception:
        # tesseract binary missing or unreadable image
        return ""


def save_temp(image: Image.Image, suffix: str = ".jpg") -> Path:
    """Persist the image to a temp file. Returns the path.

    Useful for ELA, which needs to round-trip through JPEG compression.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    image.save(tmp.name, quality=95)
    return Path(tmp.name)
