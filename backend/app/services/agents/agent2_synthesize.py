"""
Agent 2: validated entity annotations + OCR context → new patterns / rules / templates as structured JSON.
Uses LangChain ChatOllama; parses JSON into AgentArtifactEnvelope.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import (
    AGENT2_MODEL,
    DEFAULT_LLM_MODEL,
    LLM_MAX_CHARS,
    LLM_NUM_CTX,
    OLLAMA_BASE_URL,
)
from app.schemas_agents import AgentArtifactEnvelope, ValidatedEntityOcr
from app.services.agent_session_store import get_job

logger = logging.getLogger(__name__)

SYSTEM = """You are an expert extraction-schema author for banking and document OCR pipelines.
You output ONLY valid JSON. No markdown fences. The root object must have keys: patterns, rules, templates, rationale.

Rules:
- Propose NEW pattern, rule, and template objects for this document family (they may not exist in the KB yet).
- Use Python 3 `re` syntax in regex fields when applicable.
- Use snake_case for new _id fields you invent.
- If templates are too uncertain, return an empty templates array and put extraction logic in patterns and rules.
"""


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n - 20] + "\n…(truncated)"


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

    ocr_block = _truncate(ocr.full_text, LLM_MAX_CHARS)

    user_msg = f"""OCR text (primary document):
---
{ocr_block}
---

Prior retrieval notes (from Agent 1 / KB search, may be truncated):
---
{_truncate(discover_notes, 8000)}
---

User-validated entity annotations (ground truth):
---
{validated_block}
---

{f"Additional instructions: {extra_instructions}" if extra_instructions.strip() else ""}

Return a single JSON object with keys: patterns (array), rules (array), templates (array), rationale (string)."""

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=model_name,
        temperature=0.15,
        num_ctx=LLM_NUM_CTX if LLM_NUM_CTX > 0 else None,
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), ("user", "{user}")]
    )
    chain = prompt | llm
    msg = chain.invoke({"user": user_msg})
    raw_text = getattr(msg, "content", None) or str(msg)

    try:
        obj = json.loads(_extract_json_object(raw_text))
        env = AgentArtifactEnvelope.model_validate(obj)
        return env, raw_text, model_name
    except Exception as e:
        logger.exception("Agent2 JSON parse failed")
        raise ValueError(f"Agent 2 could not produce valid artifacts JSON: {e}") from e
