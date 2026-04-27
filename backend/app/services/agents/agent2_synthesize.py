"""
Agent 2: validated entity annotations + OCR context → KB-schema patterns/rules/templates.

No LangChain. Uses httpx → Ollama /api/chat directly, consistent with llm_regex.py.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.config import (
    AGENT2_MODEL,
    DEFAULT_LLM_MODEL,
    LLM_MAX_CHARS,
    LLM_NUM_CTX,
    OLLAMA_BASE_URL,
    OLLAMA_HTTP_TIMEOUT,
)
from app.schemas_agents import AgentArtifactEnvelope, ValidatedEntityOcr
from app.services.agent_session_store import get_job

logger = logging.getLogger(__name__)

SYSTEM = """You are an expert extraction-schema author for banking and document OCR pipelines.
You output ONLY valid JSON — no markdown fences, no prose. The root object must have exactly
these keys: patterns, rules, templates, rationale.

Output schema mirrors the KB node schema:

patterns  — array of objects, each with:
  _id            string   snake_case unique identifier you invent
  name           string
  type           string   "regex" | "string" | "spacy"
  regexPattern   string | null   Python 3 re syntax (double-escape backslashes in JSON)
  stringPattern  string | null
  source         string   "generated"
  extracts_entity_id  string   matches the _id of the entity below

rules  — array of objects, each with:
  _id            string   snake_case unique identifier
  name           string
  ruleType       string
  filtersJson    string   JSON-encoded filters dict
  entityJson     string   JSON-encoded {"entityId": "<entity._id>"}
  valueJson      string   JSON-encoded {"valuePattern": ["<pattern._id>"]}
  keyJson        string   JSON-encoded {"keyPattern": ["<pattern._id>"], "keyEntity": []}
  targets_entity_id  string

templates  — array of objects, each with:
  nerTemplateId  string   snake_case unique identifier
  name           string
  entity_settings  array of objects:
    stableId     string
    name         string
    ner_rules    array of objects:
      stableId          string
      nerRuleType       string
      name              string
      connectionType    string | null
      direction         string | null
      valueMatchSettingJson  string   JSON-encoded match setting
      keyMatchSettingJson    string | null

entities  — array of objects for any NEW entities needed:
  _id            string   snake_case
  name           string
  entityType     string   "single" | "compound"
  dataType       string   text | date | amount | currency | number | email | phone | address | id | other

rationale  — string explaining the choices made

Rules:
- Reuse existing KB _id values when an entity is already covered; create new ones for gaps.
- Use Python 3 re syntax only in regexPattern. Double-escape backslashes in JSON strings.
- Use snake_case for all new _id / stableId fields.
- If templates are uncertain, return an empty templates array and express logic in patterns + rules.
- Return ONLY the JSON object — no commentary, no markdown.
"""


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 20] + "\n…(truncated)"


def _extract_json_object(raw: str) -> str:
    t = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        return fence.group(1).strip()
    a = t.find("{")
    b = t.rfind("}")
    if a != -1 and b > a:
        return t[a : b + 1]
    return t


def _call_ollama(model_name: str, messages: list[dict[str, str]]) -> str:
    """Synchronous Ollama /api/chat call — consistent with how llm_regex.py calls Ollama."""
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.15, **({"num_ctx": LLM_NUM_CTX} if LLM_NUM_CTX > 0 else {})},
    }
    url = f"{OLLAMA_BASE_URL}/api/chat"
    with httpx.Client(timeout=OLLAMA_HTTP_TIMEOUT) as client:
        r = client.post(url, json=payload)
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
            raise ValueError(f"Ollama /api/chat response was not JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"Ollama /api/chat JSON was not an object: {data!r}")
        msg = data.get("message")
        if not isinstance(msg, dict):
            raise ValueError(f"Ollama /api/chat missing message: {str(data)[:1200]!r}")
        content = msg.get("content")
        if content is None:
            raise ValueError(f"Ollama /api/chat missing content: {str(data)[:1200]!r}")
        return str(content)


def run_agent2_synthesize(
    job_id: str,
    validated: list[ValidatedEntityOcr],
    model: str | None,
    extra_instructions: str,
) -> tuple[AgentArtifactEnvelope, str, str]:
    job = get_job(job_id)
    if not job:
        raise ValueError("Unknown job_id")

    ocr = job["ocr"]
    discover = job.get("last_discover") or {}
    discover_notes = str(discover.get("notes") or "")

    model_name = (model or "").strip() or (AGENT2_MODEL or "").strip() or DEFAULT_LLM_MODEL

    validated_lines = []
    for v in validated:
        validated_lines.append(
            f"- entity: {v.name} (kind={v.kind})\n"
            f"  landmark: {v.landmark}\n  label: {v.label}\n  value: {v.value}\n  hints: {v.hints}"
        )
    validated_block = "\n".join(validated_lines)

    user_msg = (
        f"OCR text (primary document):\n---\n{_truncate(ocr.full_text, LLM_MAX_CHARS)}\n---\n\n"
        f"Prior retrieval notes (from Agent 1 / KB search, may be truncated):\n---\n"
        f"{_truncate(discover_notes, 8000)}\n---\n\n"
        f"User-validated entity annotations (ground truth):\n---\n{validated_block}\n---\n\n"
        + (f"Additional instructions: {extra_instructions}\n\n" if extra_instructions.strip() else "")
        + "Return a single JSON object with keys: entities, patterns, rules, templates, rationale."
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    raw_text = _call_ollama(model_name, messages)

    try:
        obj = json.loads(_extract_json_object(raw_text))
        env = AgentArtifactEnvelope.model_validate(obj)
        return env, raw_text, model_name
    except Exception as e:
        logger.exception("Agent 2 JSON parse failed")
        raise ValueError(f"Agent 2 could not produce valid artifacts JSON: {e}") from e
