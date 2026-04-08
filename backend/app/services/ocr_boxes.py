"""
EasyOCR token boxes for annotation UI.

Given an upload_id (prefix used in backend/data/uploads), render a page at DPI,
run EasyOCR detail mode, and return token boxes + the page image (base64 PNG).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from app.config import OCR_USE_GPU, UPLOAD_DIR
from app.schemas import OcrBox, OcrBoxesResponse

_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        _easyocr_reader = easyocr.Reader(["en"], gpu=OCR_USE_GPU, verbose=False)
    return _easyocr_reader


def _find_upload_path(upload_id: str) -> Path:
    matches = list(Path(UPLOAD_DIR).glob(f"{upload_id}_*.pdf"))
    if not matches:
        matches = list(Path(UPLOAD_DIR).glob(f"{upload_id}_*"))
    if not matches:
        raise FileNotFoundError(upload_id)
    # If multiple, pick newest-ish by name ordering (uuid prefix is same); arbitrary but stable
    return sorted(matches)[-1]


@dataclass
class _Box:
    x0: float
    y0: float
    x1: float
    y1: float


def _bbox_from_easyocr_quad(quad: list[list[float]] | tuple) -> _Box:
    # quad is 4 points [[x,y],...]
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return _Box(min(xs), min(ys), max(xs), max(ys))


def _render_page_png(doc: fitz.Document, page_index0: int, dpi: int) -> tuple[bytes, np.ndarray]:
    page = doc.load_page(page_index0)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png = pix.tobytes("png")
    pil = Image.open(io.BytesIO(png)).convert("RGB")
    arr = np.array(pil)
    return png, arr


async def ocr_boxes_for_upload(upload_id: str, page: int, dpi: int) -> OcrBoxesResponse:
    path = _find_upload_path(upload_id)
    doc = fitz.open(str(path))
    try:
        page_index0 = page - 1
        if page_index0 < 0 or page_index0 >= doc.page_count:
            raise ValueError(f"page out of range (1..{doc.page_count})")
        png_bytes, arr = _render_page_png(doc, page_index0, dpi)
    finally:
        doc.close()

    h, w = int(arr.shape[0]), int(arr.shape[1])
    reader = _get_easyocr_reader()
    detailed = reader.readtext(arr, detail=1, paragraph=False)

    boxes: list[OcrBox] = []
    for i, row in enumerate(detailed):
        # EasyOCR format: [bbox, text, conf]
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        quad = row[0]
        text = str(row[1] or "").strip()
        conf = float(row[2]) if len(row) >= 3 and row[2] is not None else 1.0
        if not text:
            continue
        b = _bbox_from_easyocr_quad(quad)
        boxes.append(
            OcrBox(
                id=f"p{page}-{i}",
                text=text,
                page=page,
                x0=b.x0,
                y0=b.y0,
                x1=b.x1,
                y1=b.y1,
                conf=conf,
            )
        )

    image_base64 = base64.b64encode(png_bytes).decode("ascii")
    return OcrBoxesResponse(
        upload_id=upload_id,
        page=page,
        dpi=dpi,
        width=w,
        height=h,
        image_base64=image_base64,
        boxes=boxes,
    )

