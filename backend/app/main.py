"""
Regex Pattern Lab — FastAPI backend: PDF text + Ollama-powered regex generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    EXTRACTION_MODE,
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
    OCR_DPI,
    OCR_ENGINE,
    OCR_USE_GPU,
    OLLAMA_BASE_URL,
    OLLAMA_REFINEMENT_MODEL,
    UPLOAD_DIR,
)
from app.schemas import (
    GraphRagPreviewRequest,
    GraphRagPreviewResponse,
    RegexBatchRequest,
    RegexBatchResponse,
    RegexGenerateRequest,
    RegexGenerateResponse,
    OcrBoxesResponse,
    RegexValidateRequest,
    RegexValidateResponse,
    UploadResponse,
)
from app.schemas_agents import (
    AgentDiscoverRequest,
    AgentDiscoverResponse,
    AgentOcrUploadResponse,
    AgentSynthesizeRequest,
    AgentSynthesizeResponse,
)
from app.services.agent_session_store import create_job
from app.services.agents.agent1_discover import run_agent1_discover
from app.services.agents.agent2_synthesize import run_agent2_synthesize
from app.services.ocr_json_parser import normalize_ocr_json
from app.services.graph_rag import build_retrieval_query, run_graph_rag_safe
from app.services.llm_regex import generate_regex_patterns, refine_regex_patterns_with_llm
from app.services.pdf_extract import extract_document, save_upload
from app.services.ocr_boxes import ocr_boxes_for_upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _model_error_detail(exc: BaseException) -> str:
    """
    Build a non-empty message for API responses.
    HTTPException.__str__ can be '' when detail is empty; never return blank.
    """
    import traceback

    if isinstance(exc, HTTPException):
        d = exc.detail
        if isinstance(d, str) and d.strip():
            return d.strip()
        if d is not None and not isinstance(d, str):
            return str(d)
        return f"HTTPException status={exc.status_code} detail={d!r}"

    one = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    if one:
        return one
    return f"{type(exc).__name__}: (no message; see server logs)"


def _strip_invisible(s: str) -> str:
    """Remove ZWSP/BOM so we never build a detail that looks empty on screen."""
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        s = s.replace(ch, "")
    return s.strip()


def _graph_rag_context_for_request(
    body: RegexGenerateRequest,
) -> tuple[str | None, list[dict], str]:
    """Build KB context string for the LLM; returns (context, hits, error)."""
    if not body.use_graph_rag or not GRAPH_RAG_ENABLED:
        return None, [], ""
    names = [e.name for e in body.entities if e.name.strip()]
    hints = "\n".join((e.hints or "") for e in body.entities)
    snippet = (body.full_text or "")[:3000]
    rq = build_retrieval_query(names, hints, snippet)
    return run_graph_rag_safe(
        index_dir=GRAPH_RAG_INDEX_DIR,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        retrieval_query=rq,
        vector_k=GRAPH_RAG_VECTOR_K,
        max_context_chars=GRAPH_RAG_MAX_CONTEXT_CHARS,
    )


def _raise_model_error(exc: BaseException) -> None:
    logger.exception("LLM call failed")
    base = _strip_invisible(_model_error_detail(exc) or "")
    alt = _strip_invisible(str(exc))
    parts: list[str] = []
    if base:
        parts.append(base)
    if alt and alt != base and alt not in base:
        parts.append(alt)
    args = getattr(exc, "args", ())
    if args and str(args) not in "".join(parts):
        parts.append(f"args={args!r}")
    core = " | ".join(parts) if parts else f"{exc.__class__.__qualname__}: {exc!r}"
    core = _strip_invisible(core)
    if not core:
        core = f"{exc.__class__.__qualname__}: {exc!r}"
    if not _strip_invisible(core):
        core = f"{type(exc).__name__} (no message; check server logs)"
    msg = f"Model error: {core}"
    # Final guard: colon with nothing visible (e.g. only whitespace / odd Unicode)
    tail = msg.split(":", 1)[-1] if ":" in msg else msg
    if not _strip_invisible(tail):
        msg = f"Model error: {type(exc).__name__}: {exc!r}"
    raise HTTPException(status_code=502, detail=msg) from exc


app = FastAPI(title="Regex Pattern Lab", version="0.2.0", description="PDF/OCR regex + agentic KB discovery")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/agent/ocr-upload", response_model=AgentOcrUploadResponse)
async def agent_ocr_upload(file: Annotated[UploadFile, File()]):
    """Upload Azure-style ocr.json; returns job_id for Agent 1 / Agent 2."""
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(400, "Please upload a .json file (e.g. Azure OCR export).")

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

    jid = create_job(ocr, file.filename or "ocr.json")
    prev = ocr.full_text[:4000] + ("..." if len(ocr.full_text) > 4000 else "")
    return AgentOcrUploadResponse(
        job_id=jid,
        source_name=file.filename or "ocr.json",
        page_count=ocr.page_count,
        line_count=len(ocr.lines),
        char_count=len(ocr.full_text),
        text_preview=prev,
    )


@app.post("/api/agent/discover", response_model=AgentDiscoverResponse)
async def agent_discover(body: AgentDiscoverRequest):
    """
    Agent 1: chunk/embed OCR lines → session FAISS; per entity search KB (graph vector index) + OCR similarity.
    """
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
        logger.exception("Agent 1 discover failed")
        raise HTTPException(500, str(e)) from e


@app.post("/api/agent/synthesize", response_model=AgentSynthesizeResponse)
async def agent_synthesize(body: AgentSynthesizeRequest):
    """Agent 2: validated annotations → structured patterns / rules / templates JSON."""
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
        logger.exception("Agent 2 synthesize failed")
        _raise_model_error(e)


@app.get("/api/graph-rag/status")
def graph_rag_status():
    """Whether Graph RAG is configured and the vector index directory exists."""
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


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(
    file: Annotated[UploadFile, File()],
    extraction_mode: Annotated[str | None, Form()] = None,
    ocr_engine: Annotated[str | None, Form()] = None,
    ocr_dpi: Annotated[int | None, Form()] = None,
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    raw = await file.read()
    max_b = MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_b:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_MB} MB).")

    uid = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{uid}_{file.filename}"
    save_upload(path, raw)

    mode = (extraction_mode or EXTRACTION_MODE or "scan").lower()
    if mode not in ("scan", "auto", "embedded"):
        raise HTTPException(400, "extraction_mode must be scan, auto, or embedded.")
    engine = (ocr_engine or OCR_ENGINE or "docling").lower()
    if engine not in ("paddle", "easyocr", "docling"):
        raise HTTPException(400, "ocr_engine must be paddle, easyocr, or docling.")
    dpi = int(ocr_dpi) if ocr_dpi is not None else OCR_DPI
    if dpi < 72 or dpi > 600:
        raise HTTPException(400, "ocr_dpi must be between 72 and 600.")

    try:
        text, pages, method = extract_document(
            raw,
            mode=mode,
            ocr_engine=engine,
            use_gpu=OCR_USE_GPU,
            ocr_dpi=dpi,
        )
    except Exception as e:
        logger.exception("Extract failed")
        raise HTTPException(500, f"PDF extraction failed: {e}") from e

    preview = text[:4000] + ("..." if len(text) > 4000 else "")
    return UploadResponse(
        upload_id=uid,
        filename=file.filename,
        pages=pages,
        text_preview=preview,
        full_text=text,
        extraction_method=method,
        extraction_mode=mode,
        ocr_engine=engine,
        ocr_dpi=dpi,
    )


@app.get("/api/ocr-boxes", response_model=OcrBoxesResponse)
async def get_ocr_boxes(
    upload_id: str,
    page: int = 1,
    dpi: int = 200,
):
    """
    Return a rendered page image + OCR token boxes (EasyOCR) for annotation UI.
    Page is 1-indexed.
    """
    if page < 1:
        raise HTTPException(400, "page must be >= 1")
    if dpi < 72 or dpi > 600:
        raise HTTPException(400, "dpi must be between 72 and 600.")
    try:
        return await ocr_boxes_for_upload(upload_id=upload_id, page=page, dpi=dpi)
    except FileNotFoundError:
        raise HTTPException(404, "Upload not found. Upload the PDF again.") from None
    except Exception as e:
        logger.exception("OCR boxes failed")
        raise HTTPException(500, f"OCR boxes failed: {e}") from e


@app.post("/api/generate-regex-batch", response_model=RegexBatchResponse)
async def generate_regex_batch(body: RegexBatchRequest):
    """Run the same prompt against several Ollama models (compare generalization ideas)."""

    kb: str | None = None
    g_err = ""
    g_hits: list[dict] = []
    if body.use_graph_rag and GRAPH_RAG_ENABLED:
        rq = build_retrieval_query(
            [e.name for e in body.entities],
            "\n".join((e.hints or "") for e in body.entities),
            (body.full_text or "")[:3000],
        )
        kb, g_hits, g_err = await asyncio.to_thread(
            lambda: run_graph_rag_safe(
                index_dir=GRAPH_RAG_INDEX_DIR,
                neo4j_uri=NEO4J_URI,
                neo4j_user=NEO4J_USER,
                neo4j_password=NEO4J_PASSWORD,
                retrieval_query=rq,
                vector_k=GRAPH_RAG_VECTOR_K,
                max_context_chars=GRAPH_RAG_MAX_CONTEXT_CHARS,
            )
        )

    async def one(m: str) -> RegexGenerateResponse:
        gen = await generate_regex_patterns(
            body.full_text,
            body.entities,
            m,
            body.extra_instructions,
            [],
            kb_context=kb,
        )
        return gen.model_copy(
            update={
                "graph_rag_used": bool(body.use_graph_rag and GRAPH_RAG_ENABLED),
                "graph_rag_error": g_err if body.use_graph_rag else "",
                "graph_rag_hits": g_hits if body.use_graph_rag else [],
            }
        )

    try:
        results = await asyncio.gather(*[one(m) for m in body.models])
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch LLM failed")
        _raise_model_error(e)
    return RegexBatchResponse(results=list(results))


@app.post("/api/generate-regex", response_model=RegexGenerateResponse)
async def generate_regex(body: RegexGenerateRequest):
    if not body.entities:
        raise HTTPException(400, "At least one entity is required.")
    kb_ctx: str | None = None
    g_hits: list[dict] = []
    g_err = ""
    if body.use_graph_rag and GRAPH_RAG_ENABLED:
        kb_ctx, g_hits, g_err = await asyncio.to_thread(_graph_rag_context_for_request, body)
    try:
        gen = await generate_regex_patterns(
            body.full_text,
            body.entities,
            body.model,
            body.extra_instructions,
            body.additional_full_texts,
            kb_context=kb_ctx,
        )
        gen = gen.model_copy(
            update={
                "graph_rag_used": bool(body.use_graph_rag and GRAPH_RAG_ENABLED),
                "graph_rag_error": g_err if body.use_graph_rag else "",
                "graph_rag_hits": g_hits if body.use_graph_rag else [],
            }
        )
        ref_model = (body.refinement_model or "").strip() or OLLAMA_REFINEMENT_MODEL
        if ref_model:
            try:
                gen = await refine_regex_patterns_with_llm(
                    body.full_text,
                    body.entities,
                    gen,
                    ref_model,
                )
            except Exception as e:
                logger.warning("Regex refinement pass failed; returning first-pass patterns only: %s", e)
        return gen
    except HTTPException:
        raise
    except Exception as e:
        _raise_model_error(e)


@app.post("/api/test-regex")
async def test_regex(body: RegexGenerateRequest):
    """Generate patterns and return first match per entity on the provided text."""
    gen = await generate_regex(body)
    results: dict[str, list[str]] = {}
    flags_map = {
        "IGNORECASE": re.IGNORECASE,
        "DOTALL": re.DOTALL,
        "MULTILINE": re.MULTILINE,
    }

    for p in gen.patterns:
        if not p.pattern.strip():
            results[p.entity] = []
            continue
        fl = 0
        for part in re.split(r"[|,]", p.flags):
            part = part.strip()
            if part in flags_map:
                fl |= flags_map[part]
        try:
            m = re.findall(p.pattern, body.full_text, fl)
            if m and isinstance(m[0], tuple):
                m = [x for t in m for x in t if x]
            results[p.entity] = m[:20] if isinstance(m, list) else [str(m)][:20]
        except re.error as e:
            results[p.entity] = [f"<regex error: {e}>"]

    return {"generation": gen.model_dump(), "matches": results}


@app.post("/api/validate-regex", response_model=RegexValidateResponse)
async def validate_regex(body: RegexValidateRequest):
    """Run provided patterns against provided text and return matches/errors per entity."""
    results: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    flags_map = {
        "IGNORECASE": re.IGNORECASE,
        "DOTALL": re.DOTALL,
        "MULTILINE": re.MULTILINE,
    }
    for p in body.patterns:
        if not p.pattern.strip():
            results[p.entity] = []
            continue
        fl = 0
        for part in re.split(r"[|,]", (p.flags or "")):
            part = part.strip()
            if part in flags_map:
                fl |= flags_map[part]
        try:
            m = re.findall(p.pattern, body.full_text, fl)
            if m and isinstance(m[0], tuple):
                m = [x for t in m for x in t if x]
            results[p.entity] = m[:50] if isinstance(m, list) else [str(m)][:50]
        except re.error as e:
            results[p.entity] = []
            errors[p.entity] = str(e)
    return RegexValidateResponse(matches=results, errors=errors)
