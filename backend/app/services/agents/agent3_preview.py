"""
Agent 3: apply synthesized patterns (regex / string) against OCR lines to verify extraction.

Runs entirely locally — no LLM call needed. Reads OCR from the session store,
iterates over patterns in the AgentArtifactEnvelope, and returns match hits
grouped by entity.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.schemas_agents import (
    AgentArtifactEnvelope,
    AgentPreviewResponse,
    EntityExtractionResult,
    PatternExtractionHit,
)
from app.services.agent_session_store import get_job

logger = logging.getLogger(__name__)

_MAX_HITS_PER_PATTERN = 20


def _entity_name_from_artifacts(artifacts: AgentArtifactEnvelope, entity_id: str) -> str:
    for e in artifacts.entities:
        if e.get("_id") == entity_id:
            return str(e.get("name", entity_id))
    return entity_id


def _run_regex(pattern_str: str, ocr_lines: list[Any]) -> list[tuple[str, str, int, int]]:
    """Return list of (matched_text, line_text, page, char_offset)."""
    hits: list[tuple[str, str, int, int]] = []
    try:
        rx = re.compile(pattern_str)
    except re.error as e:
        logger.warning("Invalid regex %r: %s", pattern_str, e)
        return hits
    for ln in ocr_lines:
        text = str(ln.text or "")
        for m in rx.finditer(text):
            hits.append((m.group(0), text, int(ln.page_number), m.start()))
            if len(hits) >= _MAX_HITS_PER_PATTERN:
                return hits
    return hits


def _run_string(pattern_str: str, ocr_lines: list[Any]) -> list[tuple[str, str, int, int]]:
    """Case-insensitive substring search."""
    hits: list[tuple[str, str, int, int]] = []
    needle = pattern_str.lower()
    if not needle:
        return hits
    for ln in ocr_lines:
        text = str(ln.text or "")
        idx = text.lower().find(needle)
        if idx != -1:
            hits.append((text[idx : idx + len(needle)], text, int(ln.page_number), idx))
            if len(hits) >= _MAX_HITS_PER_PATTERN:
                return hits
    return hits


def run_agent3_preview(job_id: str, artifacts: AgentArtifactEnvelope) -> AgentPreviewResponse:
    job = get_job(job_id)
    if not job:
        return AgentPreviewResponse(job_id=job_id, results=[], error="Unknown job_id — upload OCR JSON again.")

    ocr = job["ocr"]
    lines = ocr.lines

    # Build entity_id → EntityExtractionResult map, seeded from artifacts.entities
    entity_map: dict[str, EntityExtractionResult] = {}

    # Seed from explicit entities list
    for ent in artifacts.entities:
        eid = str(ent.get("_id", ""))
        ename = str(ent.get("name", eid))
        if eid:
            entity_map[eid] = EntityExtractionResult(entity_id=eid, entity_name=ename)

    # Apply patterns
    for pat in artifacts.patterns:
        pid = str(pat.get("_id", ""))
        ptype = str(pat.get("type", "regex")).lower()
        entity_id = str(pat.get("extracts_entity_id", ""))
        regex_val = pat.get("regexPattern") or ""
        string_val = pat.get("stringPattern") or ""

        # Ensure there's an entity bucket even if not in artifacts.entities
        if entity_id not in entity_map:
            entity_map[entity_id] = EntityExtractionResult(
                entity_id=entity_id,
                entity_name=_entity_name_from_artifacts(artifacts, entity_id),
            )

        raw_hits: list[tuple[str, str, int, int]] = []

        if ptype == "regex" and regex_val:
            raw_hits = _run_regex(str(regex_val), lines)
        elif ptype == "string" and string_val:
            raw_hits = _run_string(str(string_val), lines)
        elif ptype == "spacy":
            # spaCy patterns can't be executed without a model; note it
            entity_map[entity_id].no_pattern_reason = (
                entity_map[entity_id].no_pattern_reason
                or "spaCy pattern — cannot execute locally without a NLP model"
            )
        elif not regex_val and not string_val:
            entity_map[entity_id].no_pattern_reason = (
                entity_map[entity_id].no_pattern_reason or f"Pattern '{pid}' has no regex or string value"
            )

        for matched_text, line_text, page, offset in raw_hits:
            entity_map[entity_id].hits.append(
                PatternExtractionHit(
                    pattern_id=pid,
                    pattern_type=ptype,
                    matched_text=matched_text,
                    line_text=line_text[:300],
                    page=page,
                    char_offset=offset,
                )
            )

    # Mark matched flag
    for er in entity_map.values():
        er.matched = len(er.hits) > 0

    results = list(entity_map.values())
    total = sum(len(r.hits) for r in results)

    return AgentPreviewResponse(job_id=job_id, results=results, total_hits=total)
