"""
Spatial reading order and line grouping for OCR boxes (EasyOCR-style bboxes).
Handles multi-column pages and keeps tabular rows aligned as TSV when possible.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1e-6)


def _bbox_to_xyxy(bbox: Any) -> tuple[float, float, float, float]:
    """EasyOCR bbox: list of 4 [x,y] points."""
    arr = np.asarray(bbox, dtype=float)
    if arr.size < 8:
        return 0.0, 0.0, 0.0, 0.0
    xs = arr[::2]
    ys = arr[1::2]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def parse_easyocr_detailed(
    detailed: list[tuple[Any, str, float]],
) -> list[TextBox]:
    boxes: list[TextBox] = []
    for item in detailed:
        if not item or len(item) < 3:
            continue
        bbox, text, conf = item[0], item[1], float(item[2])
        t = (text or "").strip()
        if not t:
            continue
        x0, y0, x1, y1 = _bbox_to_xyxy(bbox)
        boxes.append(TextBox(text=t, x0=x0, y0=y0, x1=x1, y1=y1, conf=conf))
    return boxes


def _median(vals: list[float], default: float = 1.0) -> float:
    if not vals:
        return default
    return float(np.median(np.asarray(vals, dtype=float)))


def cluster_into_lines(boxes: list[TextBox], line_merge_factor: float = 0.45) -> list[list[TextBox]]:
    """Group boxes onto the same visual line using y-overlap heuristic."""
    if not boxes:
        return []
    heights = [b.height for b in boxes]
    med_h = _median(heights, 12.0)
    thresh = max(med_h * line_merge_factor, 3.0)

    sorted_boxes = sorted(boxes, key=lambda b: b.y_center)
    lines: list[list[TextBox]] = []
    current: list[TextBox] = []
    current_y: float | None = None

    for b in sorted_boxes:
        if current_y is None:
            current = [b]
            current_y = b.y_center
            continue
        if abs(b.y_center - current_y) <= thresh:
            current.append(b)
            current_y = (current_y * (len(current) - 1) + b.y_center) / len(current)
        else:
            lines.append(sorted(current, key=lambda x: x.x0))
            current = [b]
            current_y = b.y_center
    if current:
        lines.append(sorted(current, key=lambda x: x.x0))
    lines.sort(key=lambda line: _median([b.y_center for b in line], 0.0))
    return lines


def _column_gap_threshold(lines: list[list[TextBox]], page_width: float) -> float:
    """Estimate min horizontal gap that suggests a new column (not just a space)."""
    gaps: list[float] = []
    for line in lines:
        for i in range(len(line) - 1):
            g = line[i + 1].x0 - line[i].x1
            if g > 2:
                gaps.append(g)
    if not gaps:
        return max(page_width * 0.04, 12.0)
    return max(_median(gaps, 8.0) * 4.0, page_width * 0.035, 14.0)


def lines_to_text(lines: list[list[TextBox]], page_width: float) -> str:
    """Join lines; use tabs when within-line gaps look tabular."""
    col_gap = _column_gap_threshold(lines, page_width)
    out_lines: list[str] = []
    for line in lines:
        if not line:
            continue
        parts: list[str] = []
        prev_x1 = line[0].x0
        for i, b in enumerate(line):
            if i > 0:
                gap = b.x0 - prev_x1
                sep = "\t" if gap >= col_gap else " "
                parts.append(sep)
            parts.append(b.text)
            prev_x1 = b.x1
        out_lines.append("".join(parts).strip())
    return "\n".join(out_lines)


def _looks_tabular(lines: list[list[TextBox]], min_rows: int = 3) -> bool:
    """Heuristic: similar column count across several rows → likely a table."""
    if len(lines) < min_rows:
        return False
    counts = [len(L) for L in lines if len(L) >= 2]
    if len(counts) < min_rows:
        return False
    med = float(np.median(counts))
    if med < 2:
        return False
    consistent = sum(1 for c in counts if abs(c - med) <= 1.5) / len(counts)
    return consistent >= 0.55


def page_text_from_easyocr_detailed(
    detailed: list[tuple[Any, str, float]],
    image_width: float,
) -> str:
    """Turn EasyOCR detail=1 output into clean reading-order text."""
    boxes = parse_easyocr_detailed(detailed)
    if not boxes:
        return ""
    lines = cluster_into_lines(boxes)
    w = image_width if image_width > 0 else max(b.x1 for b in boxes) + 1.0
    text = lines_to_text(lines, w)

    if _looks_tabular(lines):
        # reinforce TSV: split lines that used spaces between columns poorly
        text = _refine_tsv(text)
    return text


def _refine_tsv(text: str) -> str:
    """Normalize multiple spaces between tokens to tabs on table-like blocks."""
    lines = text.split("\n")
    refined: list[str] = []
    for ln in lines:
        if re.search(r" {3,}", ln):
            ln = re.sub(r" {3,}", "\t", ln)
        refined.append(ln)
    return "\n".join(refined)
