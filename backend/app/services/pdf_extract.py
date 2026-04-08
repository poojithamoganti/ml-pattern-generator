"""
PDF → clean text: layout-aware OCR (Paddle, EasyOCR) or optional Docling pipeline.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from app.services.docling_extract import extract_pdf_with_docling, is_docling_available
from app.services.reading_order import page_text_from_easyocr_detailed
from app.services.paddle_structure import is_paddle_available, ocr_page_rgb_with_paddle

logger = logging.getLogger(__name__)

_easyocr_reader = None


def _get_easyocr_reader(use_gpu: bool):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    return _easyocr_reader


def extract_with_pymupdf(pdf_bytes: bytes) -> tuple[str, int, str]:
    """Returns (full_text, page_count, method_label)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text") or "")
        text = "\n\n".join(parts)
        return text.strip(), doc.page_count, "pymupdf_embedded"
    finally:
        doc.close()


def _page_to_image_rgb(page: fitz.Page, dpi: int) -> tuple[np.ndarray, float]:
    """Render page to RGB numpy array; return (array, width in px for layout)."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil)
    w = float(arr.shape[1])
    return arr, w


def ocr_pdf_easyocr_layout(
    pdf_bytes: bytes,
    use_gpu: bool,
    dpi: int,
) -> tuple[str, int, str]:
    """EasyOCR with spatial reading order (multi-column + tabular gaps)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    reader = _get_easyocr_reader(use_gpu)
    try:
        chunks: list[str] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            arr, width = _page_to_image_rgb(page, dpi)
            detailed = reader.readtext(arr, detail=1, paragraph=False)
            page_text = page_text_from_easyocr_detailed(detailed, width)
            header = f"--- Page {i + 1} ---"
            chunks.append(f"{header}\n{page_text}".strip())
        full = "\n\n".join(chunks)
        return full.strip(), doc.page_count, f"easyocr_layout_{dpi}dpi"
    finally:
        doc.close()


def ocr_pdf_paddle_structure(
    pdf_bytes: bytes,
    use_gpu: bool,
    dpi: int,
) -> tuple[str, int, str]:
    """Paddle PP-Structure: layout + tables (HTML → TSV) + OCR."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        chunks: list[str] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            arr, _w = _page_to_image_rgb(page, dpi)
            page_text = ocr_page_rgb_with_paddle(arr, use_gpu=use_gpu)
            header = f"--- Page {i + 1} ---"
            chunks.append(f"{header}\n{page_text}".strip())
        full = "\n\n".join(chunks)
        return full.strip(), doc.page_count, f"paddle_ppstructure_{dpi}dpi"
    finally:
        doc.close()


def _run_ocr(
    pdf_bytes: bytes,
    engine: str,
    use_gpu: bool,
    dpi: int,
) -> tuple[str, int, str]:
    eng = (engine or "easyocr").lower()
    if eng == "docling":
        if not is_docling_available():
            logger.warning("Docling requested but not installed; falling back to EasyOCR layout.")
            return ocr_pdf_easyocr_layout(pdf_bytes, use_gpu, dpi)
        try:
            return extract_pdf_with_docling(pdf_bytes)
        except Exception as e:
            logger.exception("Docling conversion failed: %s", e)
            return ocr_pdf_easyocr_layout(pdf_bytes, use_gpu, dpi)
    if eng == "paddle":
        if not is_paddle_available():
            logger.warning("Paddle requested but not available; falling back to EasyOCR layout.")
            return ocr_pdf_easyocr_layout(pdf_bytes, use_gpu, dpi)
        try:
            return ocr_pdf_paddle_structure(pdf_bytes, use_gpu, dpi)
        except Exception as e:
            logger.exception("Paddle PP-Structure failed: %s", e)
            return ocr_pdf_easyocr_layout(pdf_bytes, use_gpu, dpi)
    return ocr_pdf_easyocr_layout(pdf_bytes, use_gpu, dpi)


def extract_document(
    pdf_bytes: bytes,
    *,
    mode: str = "scan",
    ocr_engine: str = "paddle",
    use_gpu: bool = True,
    ocr_dpi: int = 300,
    min_chars_per_page: int = 30,
) -> tuple[str, int, str]:
    """
    mode:
      - scan: always run layout OCR (recommended for scanned bank statements / tables).
      - auto: use embedded text only if it looks substantial per page; else OCR.
      - embedded: PyMuPDF text only (no OCR).
    ocr_engine: paddle | easyocr | docling (docling requires optional install; falls back to easyocr).
    """
    mode = (mode or "scan").lower()
    if mode == "embedded":
        return extract_with_pymupdf(pdf_bytes)

    if mode == "scan":
        return _run_ocr(pdf_bytes, ocr_engine, use_gpu, ocr_dpi)

    # auto
    text, pages, method = extract_with_pymupdf(pdf_bytes)
    if pages == 0:
        return "", 0, "empty"

    avg = len(text) / max(pages, 1)
    if avg >= min_chars_per_page and len(text) >= min_chars_per_page:
        return text, pages, method

    logger.info("Low embedded text (avg %s chars/page); running layout OCR", round(avg, 1))
    return _run_ocr(pdf_bytes, ocr_engine, use_gpu, ocr_dpi)


def save_upload(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
