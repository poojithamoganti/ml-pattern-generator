"""
Optional PaddleOCR PP-Structure: layout + table → plain text (TSV for tables).
Regions are sorted by layout bbox for reading order. See PaddleOCR 2.7 PP-Structure docs.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from typing import Any

logger = logging.getLogger(__name__)

_engine: Any = None


def _get_pp_structure(use_gpu: bool):
    global _engine
    if _engine is None:
        from paddleocr import PPStructure

        # Defaults: layout=True, table=True, ocr=True (non-table regions)
        try:
            _engine = PPStructure(show_log=False, use_gpu=use_gpu, lang="en")
        except TypeError:
            _engine = PPStructure(show_log=False, use_gpu=use_gpu)
    return _engine


def _html_table_to_tsv(html: str) -> str:
    if not html or "<table" not in html.lower():
        return unescape(re.sub(r"<[^>]+>", " ", html)).strip()

    rows: list[list[str]] = []
    for tr in re.finditer(r"<tr[^>]*>([\s\S]*?)</tr>", html, re.I):
        row_html = tr.group(1)
        cells: list[str] = []
        for td in re.finditer(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", row_html, re.I):
            cell = td.group(1)
            cell = re.sub(r"<br\s*/?>", " ", cell, flags=re.I)
            cell = re.sub(r"<[^>]+>", "", cell)
            cells.append(unescape(cell).strip())
        if cells:
            rows.append(cells)
    if not rows:
        return unescape(re.sub(r"<[^>]+>", " ", html)).strip()

    return "\n".join("\t".join(r) for r in rows)


def _region_sort_key(region: dict[str, Any]) -> tuple[float, float]:
    bbox = region.get("bbox") or [0, 0, 0, 0]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x0, y0, _x1, _y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return (y0, x0)
    return (0.0, 0.0)


def _flatten_text_res(res: Any) -> str:
    """OCR region: res is (det_boxes, rec_lines) per PaddleOCR docs."""
    if res is None:
        return ""
    if isinstance(res, str):
        return res.strip()
    if isinstance(res, dict):
        html = res.get("html")
        if html:
            return _html_table_to_tsv(html)
        for key in ("text", "content"):
            v = res.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    if isinstance(res, tuple) and len(res) >= 2:
        rec_part = res[1]
        if isinstance(rec_part, list):
            parts: list[str] = []
            for item in rec_part:
                if isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str):
                    t = item[0].strip()
                    if t:
                        parts.append(t)
                elif isinstance(item, str) and item.strip():
                    parts.append(item.strip())
            return " ".join(parts)
    return ""


def _flatten_region(region: dict[str, Any]) -> str:
    rtype = (region.get("type") or "").lower()
    res = region.get("res")

    if rtype == "table" and isinstance(res, dict):
        html = res.get("html") or ""
        if html:
            return _html_table_to_tsv(html)
        return _flatten_text_res(res)

    if rtype == "table":
        return _flatten_text_res(res)

    if rtype in ("figure", "equation", "image"):
        return ""

    return _flatten_text_res(res)


def structure_result_to_text(result: list[dict[str, Any]] | Any) -> str:
    if not result or not isinstance(result, list):
        return ""

    cleaned: list[dict[str, Any]] = []
    for region in result:
        if not isinstance(region, dict):
            continue
        r = {k: v for k, v in region.items() if k != "img"}
        cleaned.append(r)

    cleaned.sort(key=_region_sort_key)
    chunks: list[str] = []
    for region in cleaned:
        t = _flatten_region(region).strip()
        if t:
            rtype = (region.get("type") or "").lower()
            if rtype == "table":
                chunks.append(t)
            else:
                chunks.append(t)

    return "\n\n".join(chunks)


def ocr_page_rgb_with_paddle(img_rgb: np.ndarray, use_gpu: bool) -> str:
    import cv2

    engine = _get_pp_structure(use_gpu)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result = engine(bgr)
    if isinstance(result, list):
        for line in result:
            if isinstance(line, dict) and "img" in line:
                line.pop("img", None)
    return structure_result_to_text(result)


def is_paddle_available() -> bool:
    """True only if PaddleOCR 2.x PPStructure imports cleanly (often needs numpy/opencv-compatible env)."""
    try:
        from paddleocr import PPStructure  # noqa: F401

        return True
    except Exception as e:
        logger.debug("Paddle PP-Structure not available: %s", e)
        return False
