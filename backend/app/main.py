"""
Pattern Rule Gen — FastAPI backend.

Ingest:  POST /api/ingest/pdf      – upload PDF (stored for page rendering)
         POST /api/ingest/ocr      – upload Azure OCR JSON (text + bboxes)
Render:  GET  /api/ingest/page     – render PDF page to base64 PNG (pypdfium2)
Agents:  POST /api/agent/discover  – Agent 1: KB + OCR gap discovery
         POST /api/agent/synthesize – Agent 2: KB-schema artifact synthesis
Graph:   GET  /api/graph-rag/status
         POST /api/graph-rag/preview
Utility: GET  /api/models
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    GRAPH_RAG_BM25_BRANCH_K,
    GRAPH_RAG_DENSE_BRANCH_K,
    GRAPH_RAG_DOC_SNIPPET_CHARS,
    GRAPH_RAG_ENABLED,
    GRAPH_RAG_HYBRID,
    GRAPH_RAG_INDEX_DIR,
    GRAPH_RAG_MAX_CONTEXT_CHARS,
    GRAPH_RAG_RRF_K,
    GRAPH_RAG_VECTOR_K,
    MAX_UPLOAD_MB,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OLLAMA_BASE_URL,
)
from app.schemas import (
    GraphRagPreviewRequest,
    GraphRagPreviewResponse,
)
from app.schemas_agents import (
    AgentDiscoverRequest,
    AgentDiscoverResponse,
    AgentOcrUploadResponse,
    AgentPageImageResponse,
    AgentPreviewRequest,
    AgentPreviewResponse,
    AgentSynthesizeRequest,
    AgentSynthesizeResponse,
    PdfUploadResponse,
)
from app.services.agent_session_store import create_job, get_job
from app.services.agents.agent1_discover import run_agent1_discover
from app.services.agents.agent2_synthesize import run_agent2_synthesize
from app.services.agents.agent3_preview import run_agent3_preview
from app.services.ocr_json_parser import normalize_ocr_json
from app.services.graph_rag import run_graph_rag_safe
from app.services.pdf_to_images import get_page_count, render_page

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _raise_agent_error(exc: BaseException, label: str = "Agent") -> None:
    logger.exception("%s failed", label)
    msg = str(exc).strip() or f"{type(exc).__name__}: (no message; see server logs)"
    raise HTTPException(status_code=500, detail=f"{label} error: {msg}") from exc


app = FastAPI(
    title="Pattern Rule Gen",
    version="0.3.0",
    description="PDF + Azure OCR JSON ingest → agentic KB gap analysis → pattern/rule/template synthesis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Step 1 — Ingest
# ---------------------------------------------------------------------------


@app.post("/api/ingest/pdf", response_model=PdfUploadResponse)
async def ingest_pdf(file: Annotated[UploadFile, File()]):
    """
    Upload the source PDF.  No OCR is run here — the PDF is stored in the
    session so that individual pages can be rendered on demand via
    GET /api/ingest/page.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a .pdf file.")

    raw = await file.read()
    max_b = MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_b:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_MB} MB).")

    try:
        page_count = await asyncio.to_thread(get_page_count, raw)
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}") from e

    # Create a job with no OCR yet; OCR will be attached when /api/ingest/ocr is called.
    from app.services.ocr_json_parser import NormalizedOcr, OcrLine
    placeholder_ocr = NormalizedOcr(lines=[], full_text="", page_count=page_count)
    jid = create_job(
        ocr=placeholder_ocr,
        source_name=file.filename,
        pdf_bytes=raw,
        pdf_page_count=page_count,
    )
    return PdfUploadResponse(job_id=jid, filename=file.filename, page_count=page_count)


@app.post("/api/ingest/ocr", response_model=AgentOcrUploadResponse)
async def ingest_ocr(
    file: Annotated[UploadFile, File()],
    job_id: str | None = None,
):
    """
    Upload the Azure OCR JSON for a document.

    If `job_id` is provided (from a prior /api/ingest/pdf call) the OCR data
    is attached to that session so both PDF images and OCR text are available.
    If omitted, a new session is created (PDF rendering will not be available).
    """
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(400, "Please upload a .json file (Azure OCR export).")

    raw = await file.read()
    max_b = MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_b:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_MB} MB).")

    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"Invalid JSON: {e}") from e

    try:
        ocr = normalize_ocr_json(data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    if job_id:
        # Attach OCR to the existing PDF session
        job = get_job(job_id)
        if job is None:
            raise HTTPException(404, f"Session '{job_id}' not found. Upload the PDF again.")
        from app.services import agent_session_store as _store
        import threading
        with _store._lock:
            _store._jobs[job_id]["ocr"] = ocr
            _store._jobs[job_id]["source_name"] = file.filename or "ocr.json"
        jid = job_id
    else:
        jid = create_job(ocr, source_name=file.filename or "ocr.json")

    preview = ocr.full_text[:4000] + ("..." if len(ocr.full_text) > 4000 else "")
    return AgentOcrUploadResponse(
        job_id=jid,
        source_name=file.filename or "ocr.json",
        page_count=ocr.page_count,
        line_count=len(ocr.lines),
        char_count=len(ocr.full_text),
        text_preview=preview,
    )


@app.get("/api/ingest/page", response_model=AgentPageImageResponse)
async def ingest_page(
    job_id: str = Query(..., description="Session job_id from /api/ingest/pdf"),
    page: int = Query(default=1, ge=1, description="1-indexed page number"),
    dpi: int = Query(default=150, ge=72, le=300, description="Render resolution"),
):
    """
    Render a single PDF page to a base64 PNG using pypdfium2.
    The PDF must have been uploaded first via POST /api/ingest/pdf.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Session '{job_id}' not found.")

    pdf_bytes: bytes | None = job.get("pdf_bytes")
    if not pdf_bytes:
        raise HTTPException(400, "No PDF in this session. Upload the PDF via POST /api/ingest/pdf first.")

    try:
        image_b64, w, h = await asyncio.to_thread(render_page, pdf_bytes, page, dpi)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("Page render failed")
        raise HTTPException(500, f"Page render failed: {e}") from e

    total = job.get("pdf_page_count") or job["ocr"].page_count or 1
    return AgentPageImageResponse(
        job_id=job_id,
        page_num=page,
        total_pages=total,
        image_b64=image_b64,
        width_px=w,
        height_px=h,
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@app.post("/api/agent/discover", response_model=AgentDiscoverResponse)
async def agent_discover(body: AgentDiscoverRequest):
    """Agent 1: embed OCR lines → session FAISS; per-entity KB (hybrid) + OCR similarity search."""
    try:
        payload = await asyncio.to_thread(
            run_agent1_discover,
            body.job_id,
            list(body.entities),
            body.kb_vector_k,
            body.ocr_chunk_k,
        )
        return AgentDiscoverResponse.model_validate(payload)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        _raise_agent_error(e, "Agent 1 discover")


@app.post("/api/agent/preview-extraction", response_model=AgentPreviewResponse)
async def agent_preview_extraction(body: AgentPreviewRequest):
    """Agent 3: apply synthesized regex/string patterns against OCR lines to verify extraction."""
    try:
        result = await asyncio.to_thread(run_agent3_preview, body.job_id, body.artifacts)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        _raise_agent_error(e, "Agent 3 preview-extraction")


@app.post("/api/agent/synthesize", response_model=AgentSynthesizeResponse)
async def agent_synthesize(body: AgentSynthesizeRequest):
    """Agent 2: validated annotations + KB gap context → KB-schema patterns/rules/templates."""
    try:
        env, raw, model_used = await asyncio.to_thread(
            run_agent2_synthesize,
            body.job_id,
            list(body.validated),
            body.model,
            body.extra_instructions,
        )
        return AgentSynthesizeResponse(
            job_id=body.job_id,
            artifacts=env,
            raw_model_text=raw,
            ollama_model=model_used,
            error="",
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        _raise_agent_error(e, "Agent 2 synthesize")


# ---------------------------------------------------------------------------
# Graph RAG utilities
# ---------------------------------------------------------------------------


@app.get("/api/graph-rag/status")
def graph_rag_status():
    idx = GRAPH_RAG_INDEX_DIR
    has_cfg = (idx / "config.json").is_file() and (idx / "vectors.faiss").is_file()
    return {
        "enabled_flag": GRAPH_RAG_ENABLED,
        "index_dir": str(idx.resolve()),
        "index_ready": has_cfg,
        "neo4j_configured": bool(NEO4J_PASSWORD),
        "neo4j_uri": NEO4J_URI,
        "hybrid_enabled": GRAPH_RAG_HYBRID,
        "hybrid_rrf_k": GRAPH_RAG_RRF_K,
        "hybrid_dense_branch_k": GRAPH_RAG_DENSE_BRANCH_K,
        "hybrid_bm25_branch_k": GRAPH_RAG_BM25_BRANCH_K,
        "doc_snippet_chars": GRAPH_RAG_DOC_SNIPPET_CHARS,
    }


@app.post("/api/graph-rag/preview", response_model=GraphRagPreviewResponse)
def graph_rag_preview(body: GraphRagPreviewRequest):
    """Debug: run vector search (+ Neo4j expand if password set) without calling the LLM."""
    if not GRAPH_RAG_ENABLED:
        return GraphRagPreviewResponse(context="", hits=[], error="GRAPH_RAG_ENABLED is false in server env.")
    ctx, hits, err = run_graph_rag_safe(
        index_dir=GRAPH_RAG_INDEX_DIR,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        retrieval_query=body.q.strip(),
        vector_k=body.k,
        max_context_chars=GRAPH_RAG_MAX_CONTEXT_CHARS,
    )
    if err or not ctx:
        return GraphRagPreviewResponse(context="", hits=hits, error=err or "empty context")
    return GraphRagPreviewResponse(context=ctx, hits=hits, error="")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


@app.get("/api/models")
async def list_ollama_models():
    """List models available in local Ollama (for UI dropdown)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
        names = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        return {"models": sorted(set(names))}
    except Exception as e:
        logger.warning("Could not list Ollama models: %s", e)
        return {"models": [], "error": str(e)}
