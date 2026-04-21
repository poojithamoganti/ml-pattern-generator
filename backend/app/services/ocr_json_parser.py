"""
Parse Azure-style OCR JSON into pages, lines, and a flat text stream for chunking / LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OcrLine:
    page_number: int
    line_index: int
    text: str
    line_id: str
    word_ids: list[str] = field(default_factory=list)


@dataclass
class NormalizedOcr:
    lines: list[OcrLine]
    full_text: str
    page_count: int

    def line_by_id(self, lid: str) -> OcrLine | None:
        for ln in self.lines:
            if ln.line_id == lid:
                return ln
        return None


def _word_x0(w: dict[str, Any]) -> float:
    bb = w.get("boundingBox") or []
    if not bb or not isinstance(bb, list):
        return 0.0
    try:
        return float(bb[0].get("x", 0))
    except (TypeError, ValueError, KeyError):
        return 0.0


def _parse_page_words(page: dict[str, Any]) -> list[dict[str, Any]]:
    words = page.get("words") or page.get("Words") or []
    out: list[dict[str, Any]] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        text = (w.get("text") or "").strip()
        if not text:
            continue
        out.append(w)
    return out


def normalize_ocr_json(raw: Any) -> NormalizedOcr:
    """
    Accepts:
    - list of page objects [{ pageNumber, words: [...] }, ...]
    - { "pages": [ ... ] }
    - single page object
    """
    pages: list[dict[str, Any]] = []
    if raw is None:
        raise ValueError("OCR JSON is empty")
    if isinstance(raw, list):
        pages = [p for p in raw if isinstance(p, dict)]
    elif isinstance(raw, dict):
        if "pages" in raw and isinstance(raw["pages"], list):
            pages = [p for p in raw["pages"] if isinstance(p, dict)]
        elif "words" in raw or "Words" in raw:
            pages = [raw]
        else:
            # try common wrappers
            for k in ("Pages", "content", "data"):
                v = raw.get(k)
                if isinstance(v, list):
                    pages = [p for p in v if isinstance(p, dict)]
                    break
    if not pages:
        raise ValueError("No pages with words found in OCR JSON")

    lines_out: list[OcrLine] = []
    for page in pages:
        pnum = int(page.get("pageNumber") or page.get("page") or 1)
        words = _parse_page_words(page)
        by_line: dict[int, list[dict[str, Any]]] = {}
        for w in words:
            li = int(w.get("line", 0))
            by_line.setdefault(li, []).append(w)
        for li in sorted(by_line.keys()):
            bucket = by_line[li]
            bucket.sort(key=_word_x0)
            parts = [(x.get("text") or "").strip() for x in bucket]
            text = " ".join(p for p in parts if p)
            if not text:
                continue
            wids = [str(x.get("id", "")) for x in bucket if x.get("id")]
            line_id = f"{pnum}-{li}"
            lines_out.append(
                OcrLine(
                    page_number=pnum,
                    line_index=li,
                    text=text,
                    line_id=line_id,
                    word_ids=wids,
                )
            )

    full = "\n".join(x.text for x in lines_out)
    page_nums = {p.get("pageNumber") or p.get("page") or 1 for p in pages}
    pc = max(len({ln.page_number for ln in lines_out}), len(page_nums), 1)
    return NormalizedOcr(lines=lines_out, full_text=full, page_count=pc)
