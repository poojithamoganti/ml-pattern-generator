"""
Agent 1: chunk OCR → session FAISS → per-entity KB (hybrid) + OCR similarity search.

No LangChain. Uses sentence_transformers + faiss directly, consistent with graph_rag.py.
Optional short LLM summary uses httpx → Ollama, consistent with llm_regex.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.config import (
    AGENT1_MODEL,
    AGENT_LLM_SUMMARY,
    AGENT_OCR_CHUNK_K,
    DEFAULT_LLM_MODEL,
    GRAPH_RAG_ENABLED,
    GRAPH_RAG_HYBRID,
    GRAPH_RAG_INDEX_DIR,
    GRAPH_RAG_MAX_CONTEXT_CHARS,
    GRAPH_RAG_VECTOR_K,
    LLM_NUM_CTX,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OLLAMA_BASE_URL,
    OLLAMA_HTTP_TIMEOUT,
)
from app.schemas import EntitySpec
from app.schemas_agents import (
    EntityDiscoveryResult,
    KbMatchBrief,
    OcrChunkHit,
)
from app.services.agent_session_store import get_job, set_discover, set_vectorstore
from app.services.graph_rag import (
    GraphRagStore,
    build_retrieval_query,
    expand_neo4j,
    run_graph_rag_safe,
    split_retrieval_query,
)

logger = logging.getLogger(__name__)

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_ST = None  # lazy singleton — same pattern as graph_rag.py


def _get_st():
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(_EMBED_MODEL)
    return _ST


# ---------------------------------------------------------------------------
# Session FAISS (per-job OCR index)
# ---------------------------------------------------------------------------

class _SessionIndex:
    """Minimal FAISS wrapper over OCR lines — same approach as GraphRagStore."""

    def __init__(self, texts: list[str], metadata: list[dict[str, Any]]):
        import faiss

        st = _get_st()
        vecs = st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.array(vecs, dtype="float32")
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        self._index = index
        self._metadata = metadata
        self._texts = texts
        self._st = st

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        st = self._st
        qv = st.encode([query], normalize_embeddings=True, show_progress_bar=False)
        qv = np.array(qv, dtype="float32")
        scores, indices = self._index.search(qv, min(k, len(self._texts)))
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


def _build_session_index(ocr_lines) -> _SessionIndex:
    texts = [ln.text for ln in ocr_lines]
    meta = [
        {"line_id": ln.line_id, "page": ln.page_number, "chunk_id": ln.line_id}
        for ln in ocr_lines
    ]
    return _SessionIndex(texts, meta)


# ---------------------------------------------------------------------------
# KB helpers (unchanged logic, no LangChain)
# ---------------------------------------------------------------------------

def _title_from_kb_row(row: dict[str, Any]) -> str:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    for k in ("name", "title"):
        if extra.get(k):
            return str(extra[k])[:160]
    text = str(row.get("text") or "")
    line = text.split("\n")[0] if text else ""
    return (line or str(row.get("primary_id", "")))[:160]


def _kb_search(store: GraphRagStore, query: str, k: int) -> list[tuple[int, float, dict[str, Any]]]:
    store.ensure_loaded()
    if GRAPH_RAG_HYBRID:
        qf, qe = split_retrieval_query(query)
        hits = store.search_hybrid(qf, qe, k)
    else:
        hits = store.search(query, k)
    return [
        (idx, score, store._metadata[idx])
        for idx, score in hits
        if 0 <= idx < len(store._metadata)
    ]


def _expand_summary(kind: str, primary_id: str) -> str:
    if not NEO4J_PASSWORD:
        return ""
    idx = Path(GRAPH_RAG_INDEX_DIR)
    if not (idx / "config.json").is_file():
        return ""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            text = expand_neo4j(driver, kind, primary_id)
            return text[:1500] if text else ""
        finally:
            driver.close()
    except Exception as e:
        logger.debug("expand_summary skip: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Optional LLM summary (httpx → Ollama, consistent with llm_regex.py)
# ---------------------------------------------------------------------------

def _maybe_llm_summary(entity_name: str, kb_lines: list[str], ocr_excerpts: list[str]) -> str:
    if not AGENT_LLM_SUMMARY:
        parts: list[str] = []
        if kb_lines:
            parts.append("KB: " + "; ".join(kb_lines[:5]))
        if ocr_excerpts:
            parts.append("OCR: " + " | ".join(ocr_excerpts[:3]))
        return " ".join(parts)[:500]

    model_name = (AGENT1_MODEL or "").strip() or DEFAULT_LLM_MODEL
    messages = [
        {
            "role": "system",
            "content": (
                "Write one concise sentence (max 35 words) summarizing how existing KB items "
                "and OCR lines relate to extracting this entity. No markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Entity: {entity_name}\n"
                f"KB references:\n{chr(10).join(kb_lines)[:4000]}\n"
                f"Relevant OCR lines:\n{chr(10).join(ocr_excerpts)[:4000]}"
            ),
        },
    ]
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": LLM_NUM_CTX} if LLM_NUM_CTX > 0 else {},
    }
    try:
        with httpx.Client(timeout=OLLAMA_HTTP_TIMEOUT) as client:
            r = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        content = data.get("message", {}).get("content", "")
        return (content or "").strip()[:500]
    except Exception as e:
        logger.warning("Agent1 LLM summary failed: %s", e)
        return " ".join(kb_lines[:3])[:500]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent1_discover(
    job_id: str,
    entities: list[EntitySpec],
    kb_vector_k: int,
    ocr_chunk_k: int,
) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise ValueError("Unknown job_id — upload OCR JSON again.")

    ocr = job["ocr"]
    if not ocr.lines:
        raise ValueError("No lines in normalized OCR.")

    # Build / cache session index over OCR lines
    session_idx: _SessionIndex | None = job.get("vectorstore")
    if session_idx is None:
        session_idx = _build_session_index(ocr.lines)
        set_vectorstore(job_id, session_idx)

    indexed = len(ocr.lines)
    kb_store = GraphRagStore(GRAPH_RAG_INDEX_DIR)
    kb_available = GRAPH_RAG_ENABLED and kb_store.available()

    results: list[EntityDiscoveryResult] = []
    g_err = ""

    for ent in entities:
        name = ent.name.strip()
        if not name:
            continue
        hints = (ent.hints or "").strip()
        snippet = ocr.full_text[:2500]
        rq_kb = build_retrieval_query([name], hints, snippet)

        kb_matches: list[KbMatchBrief] = []
        if kb_available:
            try:
                raw_hits = _kb_search(kb_store, rq_kb, min(kb_vector_k, GRAPH_RAG_VECTOR_K + 5))
                for _i, score, row in raw_hits:
                    kind = str(row.get("kind", ""))
                    pid = str(row.get("primary_id", ""))
                    text = str(row.get("text", ""))
                    summ = _expand_summary(kind, pid) or text[:400]
                    kb_matches.append(
                        KbMatchBrief(
                            kind=kind,
                            primary_id=pid,
                            score=float(score),
                            title=_title_from_kb_row(row),
                            summary=summ[:1200],
                        )
                    )
            except Exception as e:
                g_err = str(e)
                logger.exception("KB vector search failed")

        # OCR semantic search
        q_ocr = f"{name} {ent.kind or 'text'} {hints}"[:2000]
        ocr_hits: list[OcrChunkHit] = []
        try:
            hits = session_idx.search(q_ocr, k=min(ocr_chunk_k, AGENT_OCR_CHUNK_K))
            for idx, _score in hits:
                md = session_idx._metadata[idx]
                text_excerpt = session_idx._texts[idx]
                ocr_hits.append(
                    OcrChunkHit(
                        chunk_id=str(md.get("chunk_id", md.get("line_id", ""))),
                        page=int(md.get("page", 1)),
                        text_excerpt=text_excerpt[:500],
                        relevance_note="",
                    )
                )
        except Exception as e:
            logger.warning("OCR similarity search failed: %s", e)

        kb_line_summaries = [f"[{m.kind}] {m.primary_id}: {m.title}" for m in kb_matches[:8]]
        ocr_excerpts = [h.text_excerpt for h in ocr_hits[:5]]
        brief = _maybe_llm_summary(name, kb_line_summaries, ocr_excerpts)

        results.append(
            EntityDiscoveryResult(
                entity_name=name,
                kind=(ent.kind or "text").strip() or "text",
                kb_matches=kb_matches,
                ocr_chunk_hits=ocr_hits,
                brief_summary=brief,
            )
        )

    # Full-context KB pack for Agent 2 reference
    full_ctx, _hit_dicts, err2 = run_graph_rag_safe(
        index_dir=GRAPH_RAG_INDEX_DIR,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        retrieval_query=build_retrieval_query(
            [e.name for e in entities],
            "\n".join((e.hints or "") for e in entities),
            ocr.full_text[:3000],
        ),
        vector_k=min(GRAPH_RAG_VECTOR_K, kb_vector_k + 2),
        max_context_chars=min(GRAPH_RAG_MAX_CONTEXT_CHARS, 12000),
    )
    if err2 and not g_err:
        g_err = err2

    payload = {
        "job_id": job_id,
        "ocr_chunks_indexed": indexed,
        "entities": [r.model_dump() for r in results],
        "graph_rag_error": g_err,
        "notes": (full_ctx[:2000] + "…") if full_ctx else "",
    }
    set_discover(job_id, payload)
    return payload
