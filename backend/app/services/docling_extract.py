"""
IBM Docling — layout-aware PDF conversion (optional dependency).

Install: pip install -r requirements-docling.txt

Uses Docling's default pipeline (document understanding + OCR for scans).
"""

from __future__ import annotations

import logging
import os
import tempfile

import fitz  # PyMuPDF — page count only

logger = logging.getLogger(__name__)


def is_docling_available() -> bool:
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401

        return True
    except ImportError:
        return False


def extract_pdf_with_docling(pdf_bytes: bytes) -> tuple[str, int, str]:
    """
    Convert PDF with Docling; return (full_text, page_count, method_label).
    Writes a temp file because DocumentConverter expects a path or URL.
    """
    from docling.document_converter import DocumentConverter

    pymupdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = pymupdf_doc.page_count
    finally:
        pymupdf_doc.close()

    path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(pdf_bytes)
            path = tf.name

        converter = DocumentConverter()
        result = converter.convert(path)
        doc = result.document
        # Plain text is best for downstream regex; markdown preserves more structure if needed.
        text = doc.export_to_text()
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass

    return text.strip(), page_count, "docling"
