"""
Call a local Ollama model to propose reusable regex patterns.

Uses Ollama native ``/api/chat`` with a JSON Schema in ``format`` (structured outputs)
when enabled, then validates with Pydantic. Falls back to OpenAI-compatible
``/v1/chat/completions`` + loose parsing for older Ollama or failures.

Instructor / XGrammar are optional elsewhere; this path needs no extra deps beyond Pydantic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import time

import httpx
from pydantic import ValidationError

from app.config import (
    DEFAULT_LLM_MODEL,
    LLM_MAX_CHARS,
    LLM_NUM_CTX,
    OLLAMA_BASE_URL,
    OLLAMA_HTTP_TIMEOUT,
    OLLAMA_STRUCTURED_JSON,
)
from app.schemas import EntitySpec, RegexGenerateResponse, RegexLlmEnvelope, RegexPatternItem

logger = logging.getLogger(__name__)

# Inlined JSON Schema for Ollama ``format`` (avoid ``$ref`` / ``$defs`` compatibility issues).
REGEX_LLM_FORMAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "pattern": {"type": "string"},
                    "flags": {"type": "string"},
                    "rationale": {"type": "string"},
                    "confidence_notes": {"type": "string"},
                },
                "required": ["entity", "pattern"],
            },
        }
    },
    "required": ["patterns"],
}

SYSTEM_PROMPT = """You are an expert at writing portable, reusable regular expressions for information extraction.
Rules:
- Target runtime is Python 3 `re` only (what you put in "pattern" is passed to re.compile). Not PCRE, not Perl, not JavaScript.
- FORBIDDEN in patterns (will not work): \\\\K (keep/drop match — PCRE only), (?R) recursion, branch-reset (?|...). Use capturing groups () or non-capturing (?:...) instead.
- Prefer anchor-based patterns that capture values near a label/anchor phrase.
  - If the entity has a clear label in the text (often similar to the entity name), include that label as a literal anchor in the regex.
  - Keep the distance between label and value small (typically same line; allow limited dot-leaders/spaces between them).
  - Between label and value, copy what you actually see in the Document text or in the user's examples: if the line shows a dollar sign or spaces but no colon, do not require a colon; if the document uses "Key: value", mirror that. Do not assume ": " after a label unless it appears there.
  - Avoid patterns that would match any date/amount anywhere on the page without context.
- Prefer simple patterns: word boundaries \\\\b, digit runs \\\\d+, optional groups (...)?, and labeled context like Account\\\\s*Number:\\\\s*(\\\\d{16}) when the document shows that layout.
- Prefer character classes, quantifiers, and optional groups over copying long fixed strings.
- Anchor only when it improves precision on *similar* documents (invoice-like layouts), not one-off literals.
- For DOTALL or IGNORECASE, say so in "flags" (e.g. IGNORECASE, DOTALL).
- Output must be a single JSON object: a "patterns" array. Each item: entity (exact user label), pattern, flags, rationale, confidence_notes — use "" for unused string fields.
- Each "pattern" value is a normal JSON string only. NEVER prefix with r or r" — invalid JSON. Never write pattern": r"
- In JSON, backslashes must be doubled so the parsed string is valid for re: \\\\b \\\\s \\\\d etc.
- Example pattern value: "\\\\bTotal\\\\b" or "Account\\\\s*Number:\\\\s*(\\\\d{16})"

Occurrence guidance:
- If occurrence is "single": prefer a pattern that returns ONE match on a typical page (tight anchor + tight value shape).
- If occurrence is "multiple": prefer a pattern that matches ALL intended occurrences (e.g. repeated rows), still anchored by a label/landmark or a header/row key. Avoid over-broad patterns that match unrelated values.

Multi-sample documents:
- You may see several "Sample document" blocks (different PDFs or layout variants). Produce ONE regex per entity that extracts the same logical field across those variants when possible.
- Use alternation (?:A|B), optional groups (...)?, or character classes to cover real differences (e.g. with/without colon, $ vs no currency symbol) without matching random amounts elsewhere.
- If two layouts cannot be unified safely, prefer precision over recall and explain tradeoffs in confidence_notes.
"""

USER_TEMPLATE = """Document text (one or more samples; may be truncated):
---
{text}
---

For each entity below, produce ONE primary regex pattern that would generalize to similar documents (same vendor/layout family), not only this page.
When the document text or examples show how the label connects to the value (spaces, $, comma decimals, colons), reflect that in the regex.

Entities:
{entity_block}

{extra}

Return JSON with a "patterns" array only (no markdown). Each object: entity (exact name from list), pattern, flags, rationale, confidence_notes.
Patterns must be valid Python `re` syntax (no \\\\K, no r" prefix — JSON string only)."""


def _build_entity_block(entities: list[EntitySpec]) -> str:
    lines: list[str] = []
    for i, e in enumerate(entities, 1):
        hints = (e.hints or "").strip() or "(none)"
        kind = (e.kind or "text").strip() or "text"
        occ = (getattr(e, "occurrence", None) or "single").strip().lower()
        if occ not in ("single", "multiple"):
            occ = "single"
        ex_lines: list[str] = []
        for j, ex in enumerate(getattr(e, "examples", []) or [], 1):
            if not isinstance(ex, dict):
                continue
            source = str(ex.get("source", "") or "").strip()
            landmark = str(ex.get("landmark", "") or "").strip()
            label = str(ex.get("label", "") or "").strip()
            value = str(ex.get("value", "") or "").strip()
            parts: list[str] = []
            if source:
                parts.append(f'source="{source}"')
            if landmark:
                parts.append(f'landmark="{landmark}"')
            if label:
                parts.append(f'label="{label}"')
            if value:
                parts.append(f'value="{value}"')
            if parts:
                ex_lines.append(f"   example {j}: " + " ; ".join(parts))
        multi = len(ex_lines) > 1
        lines.append(
            f"{i}. name: {e.name}\n"
            f"   expected type: {kind}\n"
            f"   occurrence: {occ}\n"
            f"   layout / position hints: {hints}"
            + (
                ("\n   Multiple OCR examples below may come from different PDFs; synthesize ONE pattern that fits all.")
                if multi
                else ""
            )
            + (("\n" + "\n".join(ex_lines)) if ex_lines else "")
        )
    return "\n".join(lines)


def _ollama_error_from_body(data: Any) -> str | None:
    """Ollama sometimes returns 200 with an `error` field instead of content."""
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if err is None:
        return None
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or json.dumps(err))
    return str(err)


def _truncate(text: str, max_chars: int | None = None) -> str:
    max_chars = max_chars if max_chars is not None else LLM_MAX_CHARS
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n[... truncated ...]\n\n" + text[-tail:]


def _combine_document_samples(
    primary: str,
    additional: list[str] | None,
    max_total: int | None = None,
) -> str:
    """
    Merge primary OCR text with optional extra samples so the model sees multiple layouts.
    Splits the character budget across chunks so each variant gets space in context.
    """
    max_total = max_total if max_total is not None else LLM_MAX_CHARS
    extras = [t.strip() for t in (additional or []) if t and str(t).strip()]
    if not extras:
        return _truncate(primary.strip(), max_total)
    n = 1 + len(extras)
    per_doc = max(2000, max_total // n)
    chunks: list[str] = [_truncate(primary.strip(), per_doc)]
    for t in extras:
        chunks.append(_truncate(t.strip(), per_doc))
    parts: list[str] = [
        f"--- Sample document 1 / {len(chunks)} (primary) ---\n{chunks[0]}",
    ]
    for i, c in enumerate(chunks[1:], start=2):
        parts.append(f"\n\n--- Sample document {i} / {len(chunks)} (variant layout) ---\n{c}")
    combined = "".join(parts)
    if len(combined) > max_total:
        combined = _truncate(combined, max_total)
    return combined


def _extract_json_string(raw: str) -> str:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair_llm_json(text: str) -> str:
    """Strip invalid r-prefix before pattern strings so json.loads can succeed."""
    # "pattern": r"..." or "pattern": r "..."
    text = re.sub(r'("pattern"\s*:\s*)r\s*"', r'\1"', text, flags=re.IGNORECASE)
    return text


def _annotate_python_re_warnings(items: list[RegexPatternItem]) -> list[RegexPatternItem]:
    """Append notes when the model emitted constructs Python's re module does not support."""
    out: list[RegexPatternItem] = []
    for it in items:
        p = it.pattern
        notes: list[str] = []
        if "\\K" in p:
            notes.append(
                "Contains \\K (PCRE-only; Python `re` does not support it). Use a capture group or look-behind instead."
            )
        if "(?R)" in p or "(?0)" in p:
            notes.append("Recursive (?R) is not supported by Python `re`.")
        if not notes:
            out.append(it)
            continue
        suffix = " ".join(notes)
        cn = (it.confidence_notes or "").strip()
        merged = f"{cn} {suffix}".strip() if cn else suffix
        out.append(it.model_copy(update={"confidence_notes": merged}))
    return out


def _anchor_words_for_entity(e: EntitySpec) -> set[str]:
    """
    Extract anchor words from entity name and any quoted phrases in hints.
    These are used to reject patterns that are too broad (e.g. 'any date anywhere').
    """
    words: set[str] = set()
    # Entity name words
    for w in re.findall(r"[A-Za-z]{3,}", e.name or ""):
        words.add(w.lower())
    # Quoted phrases in hints: "..." or '...'
    for m in re.finditer(r"['\"]([^'\"]{3,})['\"]", e.hints or ""):
        for w in re.findall(r"[A-Za-z]{3,}", m.group(1)):
            words.add(w.lower())
    # Example landmark/label phrases
    for ex in getattr(e, "examples", []) or []:
        if not isinstance(ex, dict):
            continue
        for key in ("landmark", "label"):
            val = str(ex.get(key, "") or "")
            for w in re.findall(r"[A-Za-z]{3,}", val):
                words.add(w.lower())
    return words


def _pattern_has_anchor(pattern: str, anchor_words: set[str]) -> bool:
    """
    Heuristic: require at least one readable anchor token to prevent global matches.
    - If the pattern contains no letters at all, it's almost certainly unanchored.
    - If we have anchor_words, require intersection with literal-ish tokens in the pattern.
    """
    if not pattern or not pattern.strip():
        return False
    if not re.search(r"[A-Za-z]", pattern):
        return False
    # Extract alphabetic tokens present in the regex itself.
    toks = {t.lower() for t in re.findall(r"[A-Za-z]{3,}", pattern)}
    if not toks:
        return False
    if not anchor_words:
        # No anchor hint available; at least having letters is better than pure numeric patterns.
        return True
    return bool(toks & anchor_words)


def _enforce_anchor_based(
    patterns: list[RegexPatternItem], entities: list[EntitySpec]
) -> list[RegexPatternItem]:
    """
    Ensure each entity pattern is anchored to nearby label text.
    If a pattern is too broad, blank it and explain what anchor words are needed.
    """
    by_name = {e.name: e for e in entities}
    out: list[RegexPatternItem] = []
    for p in patterns:
        e = by_name.get(p.entity)
        if e is None:
            out.append(p)
            continue
        anchors = _anchor_words_for_entity(e)
        if _pattern_has_anchor(p.pattern, anchors):
            out.append(p)
            continue
        want = ", ".join(sorted(list(anchors))[:6])
        note = (
            f"Too broad / missing anchor. Include label words near the value"
            + (f" (e.g. {want})." if want else ". Add quoted label text in hints.")
        )
        out.append(
            p.model_copy(
                update={
                    "pattern": "",
                    "flags": "",
                    "rationale": "Rejected: pattern was not anchor-based.",
                    "confidence_notes": note,
                }
            )
        )
    return out


def _merge_entity_patterns(
    from_model: list[RegexPatternItem],
    entities: list[EntitySpec],
    missing_rationale: str = "Not returned by model or name mismatch.",
) -> list[RegexPatternItem]:
    names = {e.name for e in entities}
    by_entity: dict[str, RegexPatternItem] = {}
    for p in from_model:
        if p.entity in names:
            by_entity[p.entity] = p
    out: list[RegexPatternItem] = []
    for e in entities:
        if e.name in by_entity:
            out.append(by_entity[e.name])
        else:
            out.append(
                RegexPatternItem(
                    entity=e.name,
                    pattern="",
                    flags="",
                    rationale=missing_rationale,
                    confidence_notes="",
                )
            )
    return out


def _empty_patterns(entities: list[EntitySpec], rationale: str) -> list[RegexPatternItem]:
    return [
        RegexPatternItem(
            entity=e.name,
            pattern="",
            flags="",
            rationale=rationale,
            confidence_notes="",
        )
        for e in entities
    ]


def _parse_patterns_loose_dict(text: str, entities: list[EntitySpec]) -> list[RegexPatternItem]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed, returning empty patterns")
        return _empty_patterns(entities, "Model output was not valid JSON; see raw response.")

    if not isinstance(obj, dict):
        return _empty_patterns(entities, "Model JSON root was not an object; see raw response.")

    items: list[RegexPatternItem] = []
    names = {e.name for e in entities}
    for row in obj.get("patterns", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("entity", "")).strip()
        if name not in names:
            continue
        items.append(
            RegexPatternItem(
                entity=name,
                pattern=str(row.get("pattern", "")),
                flags=str(row.get("flags", "") or ""),
                rationale=str(row.get("rationale", "") or ""),
                confidence_notes=str(row.get("confidence_notes", "") or ""),
            )
        )
    return _merge_entity_patterns(items, entities)


def _parse_patterns_from_raw(raw: str, entities: list[EntitySpec]) -> list[RegexPatternItem]:
    """Prefer Pydantic validation; fall back to loose dict parsing (legacy models / fences)."""
    text = _repair_llm_json(_extract_json_string(raw))
    try:
        env = RegexLlmEnvelope.model_validate_json(text)
        merged = _merge_entity_patterns(list(env.patterns), entities)
        return _annotate_python_re_warnings(merged)
    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Pydantic envelope parse failed, using loose parser: %s", e)
    loose = _parse_patterns_loose_dict(text, entities)
    return _annotate_python_re_warnings(loose)


async def _fetch_native_structured(
    client: httpx.AsyncClient,
    url: str,
    model_name: str,
    messages: list[dict[str, str]],
    opts: dict[str, Any],
) -> str:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": opts,
        "format": REGEX_LLM_FORMAT_SCHEMA,
    }
    r = await client.post(url, json=payload)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:4000]
        raise ValueError(
            f"Ollama HTTP {e.response.status_code} at {url} (structured /api/chat): {body or str(e)}"
        ) from e
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"Ollama /api/chat response was not JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Ollama /api/chat JSON was not an object: {data!r}")

    err_msg = _ollama_error_from_body(data)
    if err_msg:
        raise ValueError(f"Ollama: {err_msg}")

    msg = data.get("message")
    if not isinstance(msg, dict):
        raise ValueError(f"Ollama /api/chat missing message: {str(data)[:1200]!r}")
    content = msg.get("content")
    if content is None:
        raise ValueError(f"Ollama /api/chat missing content: {str(data)[:1200]!r}")
    return str(content)


async def _fetch_openai_compat(
    client: httpx.AsyncClient,
    url: str,
    model_name: str,
    messages: list[dict[str, str]],
    opts: dict[str, Any],
) -> str:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": opts,
    }
    r = await client.post(url, json=payload)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:4000]
        raise ValueError(
            f"Ollama HTTP {e.response.status_code} at {url}: {body or str(e)}"
        ) from e
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raw_txt = (r.text or "")[:4000]
        raise ValueError(
            f"Ollama response was not JSON ({e}). Body starts with: {raw_txt[:500]!r}"
        ) from e
    if not isinstance(data, dict):
        raise ValueError(f"Ollama JSON was not an object: {data!r}")

    err_msg = _ollama_error_from_body(data)
    if err_msg:
        raise ValueError(f"Ollama: {err_msg}")

    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        snippet = str(data)[:1500]
        raise ValueError(
            f"Ollama returned no choices (model pulled? `ollama pull <name>`). "
            f"URL {url}. JSON keys: {keys}. Snippet: {snippet!r}"
        )
    c0 = choices[0]
    msg = c0.get("message") if isinstance(c0, dict) else None
    if not isinstance(msg, dict):
        raise ValueError(f"Ollama choice[0] has no message object: {str(data)[:1200]!r}")
    raw_val = msg.get("content")
    if raw_val is None:
        raise ValueError(f"Ollama message has no content: {str(data)[:1200]!r}")
    return str(raw_val)


async def generate_regex_patterns(
    full_text: str,
    entities: list[EntitySpec],
    model: str | None,
    extra_instructions: str,
    additional_full_texts: list[str] | None = None,
) -> RegexGenerateResponse:
    model_name = model or DEFAULT_LLM_MODEL
    text = _combine_document_samples(full_text, additional_full_texts)
    entity_block = _build_entity_block(entities)
    extra = ""
    if extra_instructions.strip():
        extra = "Additional user instructions:\n" + extra_instructions.strip()

    user_content = USER_TEMPLATE.format(
        text=text,
        entity_block=entity_block,
        extra=extra,
    )

    opts: dict[str, Any] = {"temperature": 0.1}
    if LLM_NUM_CTX > 0:
        opts["num_ctx"] = LLM_NUM_CTX

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    native_url = f"{OLLAMA_BASE_URL}/api/chat"
    openai_url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    t0 = time.perf_counter()
    raw = ""
    transport = "openai"

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_HTTP_TIMEOUT) as client:
            if OLLAMA_STRUCTURED_JSON:
                try:
                    raw = await _fetch_native_structured(
                        client, native_url, model_name, messages, opts
                    )
                    transport = "native+schema"
                    logger.info(
                        "Ollama structured /api/chat ok model=%s in %.1fs",
                        model_name,
                        time.perf_counter() - t0,
                    )
                except Exception as e:
                    logger.warning(
                        "Ollama structured /api/chat failed; falling back to OpenAI API: %s",
                        e,
                    )
            if not raw.strip():
                raw = await _fetch_openai_compat(
                    client, openai_url, model_name, messages, opts
                )
                transport = "openai"
                logger.info(
                    "Ollama OpenAI-compatible request done model=%s transport=%s in %.1fs",
                    model_name,
                    transport,
                    time.perf_counter() - t0,
                )
    except httpx.RequestError as e:
        raise ValueError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL} (is the app running?). {type(e).__name__}: {e!r}"
        ) from e

    patterns = _parse_patterns_from_raw(raw, entities)
    patterns = _enforce_anchor_based(patterns, entities)
    return RegexGenerateResponse(
        patterns=patterns,
        raw_model_text=raw,
        ollama_model=model_name,
    )
