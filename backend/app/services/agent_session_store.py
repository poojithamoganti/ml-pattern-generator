"""
In-memory job store for agentic OCR sessions (normalized OCR + optional LangChain FAISS index).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from app.services.ocr_json_parser import NormalizedOcr

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_TTL_SEC = 3600 * 6
_MAX_JOBS = 80


def _prune() -> None:
    now = time.time()
    dead = [k for k, v in _jobs.items() if now - v.get("_ts", 0) > _TTL_SEC]
    for k in dead:
        del _jobs[k]
    if len(_jobs) > _MAX_JOBS:
        # drop oldest
        items = sorted(_jobs.items(), key=lambda kv: kv[1].get("_ts", 0))
        for k, _ in items[: len(_jobs) - _MAX_JOBS + 10]:
            _jobs.pop(k, None)


def create_job(
    ocr: NormalizedOcr,
    source_name: str = "ocr.json",
    pdf_bytes: bytes | None = None,
    pdf_page_count: int = 0,
) -> str:
    jid = str(uuid.uuid4())
    with _lock:
        _prune()
        _jobs[jid] = {
            "_ts": time.time(),
            "ocr": ocr,
            "source_name": source_name,
            # Raw PDF bytes for page-image rendering (may be None if not uploaded)
            "pdf_bytes": pdf_bytes,
            "pdf_page_count": pdf_page_count,
            "vectorstore": None,
            "last_discover": None,
        }
    return jid


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        j = _jobs.get(job_id)
        if j:
            j["_ts"] = time.time()
        return j


def set_vectorstore(job_id: str, vs: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["vectorstore"] = vs
            _jobs[job_id]["_ts"] = time.time()


def set_discover(job_id: str, payload: dict[str, Any]) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["last_discover"] = payload
            _jobs[job_id]["_ts"] = time.time()
