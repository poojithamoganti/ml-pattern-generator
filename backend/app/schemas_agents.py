"""Pydantic models for agentic OCR → discover → synthesize workflow."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas import EntitySpec


class KbMatchBrief(BaseModel):
    """One KB row suggested for an entity."""

    kind: str = Field(description="entity | pattern | rule | template | ner_rule")
    primary_id: str
    score: float = 0.0
    title: str = ""
    summary: str = Field(default="", description="Short human-readable summary from graph expansion")


class OcrChunkHit(BaseModel):
    chunk_id: str = ""
    page: int = 1
    text_excerpt: str = ""
    relevance_note: str = ""


class EntityDiscoveryResult(BaseModel):
    entity_name: str
    kind: str = "text"
    kb_matches: list[KbMatchBrief] = Field(default_factory=list)
    ocr_chunk_hits: list[OcrChunkHit] = Field(default_factory=list)
    brief_summary: str = Field(
        default="",
        description="One or two sentences: what existing KB artifacts relate to this entity",
    )


class AgentOcrUploadResponse(BaseModel):
    job_id: str
    source_name: str
    page_count: int
    line_count: int
    char_count: int
    text_preview: str


class AgentDiscoverRequest(BaseModel):
    job_id: str = Field(..., min_length=8)
    entities: list[EntitySpec] = Field(..., min_length=1)
    kb_vector_k: int = Field(default=10, ge=3, le=30)
    ocr_chunk_k: int = Field(default=6, ge=1, le=20)


class AgentDiscoverResponse(BaseModel):
    job_id: str
    ocr_chunks_indexed: int
    entities: list[EntityDiscoveryResult]
    graph_rag_error: str = ""
    notes: str = ""


class ValidatedEntityOcr(BaseModel):
    """User-confirmed labels/values after reviewing Agent 1 output."""

    name: str
    kind: str = "text"
    landmark: str = ""
    label: str = ""
    value: str = ""
    hints: str = ""


class AgentSynthesizeRequest(BaseModel):
    job_id: str = Field(..., min_length=8)
    validated: list[ValidatedEntityOcr] = Field(..., min_length=1)
    model: str | None = Field(default=None, description="Ollama model for artifact JSON")
    extra_instructions: str = ""


class AgentArtifactEnvelope(BaseModel):
    """Structured output from Agent 2 — mirrors your KB JSON shapes loosely."""

    patterns: list[dict[str, Any]] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)
    templates: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""


class AgentSynthesizeResponse(BaseModel):
    job_id: str
    artifacts: AgentArtifactEnvelope
    raw_model_text: str = ""
    ollama_model: str = ""
    error: str = ""
