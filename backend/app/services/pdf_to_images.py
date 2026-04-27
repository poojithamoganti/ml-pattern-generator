"""
PDF → page images using pypdfium2 (pure rendering, no OCR).
Used for displaying annotated pages in the UI alongside Azure OCR JSON data.
"""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)


def get_page_count(pdf_bytes: bytes) -> int:
    """Return number of pages in the PDF."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    return len(doc)


def render_page(
    pdf_bytes: bytes,
    page_num: int = 1,
    dpi: int = 150,
) -> tuple[str, int, int]:
    """
    Render a single PDF page (1-indexed) to a base64-encoded PNG.

    Returns:
        (image_b64, width_px, height_px)

    Raises:
        ValueError: if page_num is out of range.
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    total = len(doc)

    if page_num < 1 or page_num > total:
        raise ValueError(f"Page {page_num} out of range (PDF has {total} page(s)).")

    page = doc[page_num - 1]  # pypdfium2 is 0-indexed
    scale = dpi / 72.0

    bitmap = page.render(scale=scale, rotation=0)
    pil_image = bitmap.to_pil()

    width_px, height_px = pil_image.size

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG", optimize=True)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    logger.debug(
        "Rendered PDF page %d/%d at %d DPI → %dx%d px",
        page_num, total, dpi, width_px, height_px,
    )
    return image_b64, width_px, height_px


def render_all_pages(
    pdf_bytes: bytes,
    dpi: int = 150,
) -> list[tuple[str, int, int]]:
    """
    Render every page of the PDF.

    Returns:
        List of (image_b64, width_px, height_px) per page (1-indexed order).
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    results: list[tuple[str, int, int]] = []
    scale = dpi / 72.0

    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        w, h = pil_image.size
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG", optimize=True)
        results.append((base64.b64encode(buf.getvalue()).decode(), w, h))

    return results
