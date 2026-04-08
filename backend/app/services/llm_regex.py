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
  - Anchors should come from the **user's example lines** (label=, landmark=) and document text — the entity **name** is only a display title; another PDF may use a different phrase (e.g. "Statement Balance as of …" instead of "New Balance"). Do **not** assume the entity name appears verbatim in OCR.
  - Keep the distance between label and value small (typically same line; allow limited dot-leaders/spaces between them).
  - Between label and value, copy what you actually see in the Document text or in the user's examples: if the line shows a dollar sign or spaces but no colon, do not require a colon; if the document uses "Key: value", mirror that. Do not assume ": " after a label unless it appears there. If one example has `$` before the amount and another does not, use optional `\\\\$?` (or similar) so both match.
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

Multiple OCR examples on ONE entity (several example lines with different label= or landmark=):
- If **label=** strings **differ** between examples (e.g. one says "New Balance", another "Statement Balance as of …"), a regex that only matches **one** of those phrases is **wrong**. You must use **alternation** of the distinct label/landmark phrases (or a shared parent landmark) so that the **value** from **each** example would be captured. Escaping: quote minimal distinctive substrings; use IGNORECASE if casing varies.
- If **landmark=** differs between examples (e.g. "Summary of Account Activity" vs "Payment Information"), the layouts are different: consider alternation that includes **both** landmark+label+value paths, or a regex whose anchors reflect each pair — do not silently drop a landmark from a non-primary example.
- In rationale or confidence_notes, briefly state that the pattern covers each listed source/example (or explain which layout is out of scope).

Bbox / annotate picks (user drew boxes on the page):
- The UI sends landmark/label/value **strings** from OCR boxes. Flattened document text is often **one long stream**; reading order may place **other words between** the label and the value even when they look adjacent on the image.
- A pattern that is only label-then-whitespace-then-value is often **wrong** unless a **Focused OCR slice** below shows that exact adjacency.
- When the slice shows filler between label and value, use a **conservative** non-greedy gap between literals and the capture (e.g. `[\\s\\S]{0,200}?` — raise the number only as needed from what you see, stay under ~500). Or anchor **landmark** then **label** then value. Set `flags` to include DOTALL if newlines appear in the gap. Avoid greedy `.*` spanning the whole page.
- If landmark and label both appear in the slice, prefer ordering them as in the slice before the value capture.
"""

USER_TEMPLATE = """Document text (one or more samples; may be truncated):
---
{text}
---
{annotation_focus}
For each entity below, produce ONE primary regex pattern that would generalize to similar documents (same vendor/layout family), not only this page.
When the document text or examples show how the label connects to the value (spaces, $, comma decimals, colons), reflect that in the regex.
If an entity lists multiple examples with **different** label= text, your single pattern must match **all** of those label→value shapes (use (?:...|...) over labels or landmarks); do not output a pattern that only fits the first example.

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
                (
                    "\n   Multiple OCR examples below may come from different PDFs. "
                    "If the label= (or landmark=) strings are NOT the same across examples, "
                    "use alternation so EACH example's value shape is reachable — do not anchor only on the entity display name, "
                    "only the first example's label, or only the first example's landmark."
                )
                if multi
                else ""
            )
            + (("\n" + "\n".join(ex_lines)) if ex_lines else "")
        )
    return "\n".join(lines)


def _local_snippet_for_phrases(
    text: str,
    phrases: list[str],
    *,
    padding: int = 160,
    max_span: int = 1200,
) -> str | None:
    """
    Smallest span of ``text`` covering first occurrence of each given phrase (bbox OCR strings),
    with padding. Used so the LLM sees real linear order between landmark/label/value.
    """
    text = text or ""
    spans: list[tuple[int, int]] = []
    for p in phrases:
        p = (p or "").strip()
        if len(p) < 2:
            continue
        i = text.find(p)
        if i >= 0:
            spans.append((i, i + len(p)))
    if not spans:
        return None
    lo = max(0, min(s[0] for s in spans) - padding)
    hi = min(len(text), max(s[1] for s in spans) + padding)
    if hi - lo > max_span:
        mid = (lo + hi) // 2
        half = max_span // 2
        lo = max(0, mid - half)
        hi = min(len(text), lo + max_span)
    return text[lo:hi]


def _build_annotation_focus_sections(
    primary: str,
    additional: list[str] | None,
    entities: list[EntitySpec],
) -> str:
    """
    For each saved bbox example, attach a short slice of the document where all non-empty
    phrases appear (primary first, then first additional doc that contains them).
    """
    add = [t.strip() for t in (additional or []) if t and t.strip()]
    chunks: list[str] = []
    for e in entities:
        examples = getattr(e, "examples", None) or []
        for k, ex in enumerate(examples, 1):
            if not isinstance(ex, dict):
                continue
            phrases: list[str] = []
            for key in ("landmark", "label", "value"):
                v = ex.get(key)
                if v and str(v).strip():
                    phrases.append(str(v).strip())
            if not phrases:
                continue
            seen: set[str] = set()
            uniq: list[str] = []
            for p in phrases:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            phrases = uniq
            placed = False
            for blob, blabel in [(primary, "primary document")] + [
                (t, f"additional document {i + 1}") for i, t in enumerate(add)
            ]:
                if not blob:
                    continue
                sn = _local_snippet_for_phrases(blob, phrases)
                if sn:
                    chunks.append(
                        f"--- Focus slice for «{e.name}» (OCR example {k}; {blabel}) ---\n{sn}"
                    )
                    placed = True
                    break
            if not placed:
                logger.debug(
                    "No focus slice for entity %r example %s (phrases not found in any doc)",
                    e.name,
                    k,
                )
    if not chunks:
        return ""
    intro = (
        "Focused OCR slices (linear text around your bbox phrase hits). "
        "Check filler between landmark / label / value here — not only the examples list.\n\n"
    )
    return "\n" + intro + "\n\n".join(chunks) + "\n\n"


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


REFINEMENT_SYSTEM_PROMPT = """You are a regex repair assistant for Python 3 `re` (not PCRE).

You receive first-pass patterns, primary OCR text, and real `re.findall` results plus any compile errors.

Output: one JSON object with key "patterns" (array). Each item: entity (exact name from the user list), pattern, flags, rationale, confidence_notes.

Rules:
- Patterns must compile with Python `re`. Forbidden: \\\\K, (?R). Pattern values are JSON strings with doubled backslashes.
- If findall already returns sensible captures, you may keep patterns unchanged or tighten only if clearly over-broad.
- If findall is empty or regex errored, propose fixes that plausibly extract the intended values from this OCR text (anchor near labels/landmarks; avoid matching random dates/amounts globally).
- Briefly note what you changed in rationale or confidence_notes.
"""


REFINEMENT_USER_TEMPLATE = """Primary document OCR (truncated; this is the same string used for findall):
---
{ocr}
---

First-pass patterns (JSON):
{first_patterns_json}

Python `re` evaluation on that OCR:
Matches per entity (findall):
{matches_json}

Compile/runtime errors:
{errors_json}

Entity names (output patterns[].entity must match exactly):
{names_list}

Return JSON only: an object with a "patterns" array (same shape as above)."""


def evaluate_patterns_on_ocr(
    full_text: str, items: list[RegexPatternItem]
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Run the same logic as /api/validate-regex for use in the refinement pass."""
    flags_map = {
        "IGNORECASE": re.IGNORECASE,
        "DOTALL": re.DOTALL,
        "MULTILINE": re.MULTILINE,
    }
    matches: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for p in items:
        if not p.pattern.strip():
            matches[p.entity] = []
            continue
        fl = 0
        for part in re.split(r"[|,]", (p.flags or "")):
            part = part.strip()
            if part in flags_map:
                fl |= flags_map[part]
        try:
            m = re.findall(p.pattern, full_text, fl)
            if m and isinstance(m[0], tuple):
                m = [x for t in m for x in t if x]
            matches[p.entity] = m[:40] if isinstance(m, list) else [str(m)][:40]
        except re.error as e:
            matches[p.entity] = []
            errors[p.entity] = str(e)
    return matches, errors


async def refine_regex_patterns_with_llm(
    primary_ocr: str,
    entities: list[EntitySpec],
    first_pass: RegexGenerateResponse,
    refinement_model: str,
) -> RegexGenerateResponse:
    """
    Second LLM pass: sees actual Python match results on primary OCR and may repair patterns.
    """
    ocr = _truncate(primary_ocr.strip(), LLM_MAX_CHARS)
    matches, errors = evaluate_patterns_on_ocr(primary_ocr, first_pass.patterns)
    first_json = json.dumps(
        [p.model_dump() for p in first_pass.patterns],
        ensure_ascii=False,
        indent=2,
    )
    matches_json = json.dumps(matches, ensure_ascii=False, indent=2)
    errors_json = json.dumps(errors, ensure_ascii=False, indent=2)
    names_list = ", ".join(repr(e.name) for e in entities)
    user_content = REFINEMENT_USER_TEMPLATE.format(
        ocr=ocr,
        first_patterns_json=first_json,
        matches_json=matches_json,
        errors_json=errors_json,
        names_list=names_list,
    )
    opts: dict[str, Any] = {"temperature": 0.05}
    if LLM_NUM_CTX > 0:
        opts["num_ctx"] = LLM_NUM_CTX
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    native_url = f"{OLLAMA_BASE_URL}/api/chat"
    openai_url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    raw = ""
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_HTTP_TIMEOUT) as client:
            if OLLAMA_STRUCTURED_JSON:
                try:
                    raw = await _fetch_native_structured(
                        client, native_url, refinement_model, messages, opts
                    )
                    logger.info(
                        "Refinement Ollama structured ok model=%s in %.1fs",
                        refinement_model,
                        time.perf_counter() - t0,
                    )
                except Exception as e:
                    logger.warning("Refinement structured /api/chat failed: %s", e)
            if not raw.strip():
                raw = await _fetch_openai_compat(
                    client, openai_url, refinement_model, messages, opts
                )
                logger.info(
                    "Refinement Ollama OpenAI API model=%s in %.1fs",
                    refinement_model,
                    time.perf_counter() - t0,
                )
    except httpx.RequestError as e:
        raise ValueError(
            f"Refinement: cannot reach Ollama at {OLLAMA_BASE_URL}. {type(e).__name__}: {e!r}"
        ) from e

    refined_list = _parse_patterns_from_raw(raw, entities)
    refined_list = _enforce_anchor_based(refined_list, entities)
    return RegexGenerateResponse(
        patterns=refined_list,
        raw_model_text=first_pass.raw_model_text,
        ollama_model=first_pass.ollama_model,
        refinement_raw_model_text=raw,
        refinement_model=refinement_model,
    )


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
    annotation_focus = _build_annotation_focus_sections(
        full_text.strip(),
        additional_full_texts,
        entities,
    )
    entity_block = _build_entity_block(entities)
    extra = ""
    if extra_instructions.strip():
        extra = "Additional user instructions:\n" + extra_instructions.strip()

    user_content = USER_TEMPLATE.format(
        text=text,
        annotation_focus=annotation_focus,
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
