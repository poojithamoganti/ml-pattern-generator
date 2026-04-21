"""Runtime configuration (env overrides)."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", Path(__file__).resolve().parent.parent / "data" / "uploads"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Ollama OpenAI-compatible API
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
# Second pass: Python findall results + OCR → repair regex. Default qwen2.5:7b (strong instruction-following,
# good for structured JSON / code-like fixes; different family from typical llama generator). Override via env or set to "" to disable.
OLLAMA_REFINEMENT_MODEL = os.getenv("OLLAMA_REFINEMENT_MODEL", "qwen2.5:7b").strip()

# Sent to Ollama as prompt context cap (smaller = faster; full PDF still in UI, LLM sees truncated slice)
LLM_MAX_CHARS = int(os.getenv("LLM_MAX_CHARS", "12000"))
# Ollama option: context window for this request (lower can speed CPU inference)
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))
OLLAMA_HTTP_TIMEOUT = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "900"))
# Use Ollama /api/chat + JSON Schema in `format` (structured outputs); falls back if request fails.
OLLAMA_STRUCTURED_JSON = os.getenv("OLLAMA_STRUCTURED_JSON", "1").lower() in ("1", "true", "yes")

# Prefer GPU for EasyOCR / Paddle when available
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "1").lower() in ("1", "true", "yes")

# scan: always layout OCR (best for scanned bank statements / tables)
# auto: use embedded PDF text if dense, else OCR
# embedded: PyMuPDF text only
EXTRACTION_MODE = os.getenv("EXTRACTION_MODE", "scan").lower()

# paddle: PP-Structure (layout + tables) — needs a clean env; see requirements-paddle.txt
# easyocr: spatial reading order + tab gaps (works in typical torch/conda stacks)
# docling: IBM Docling pipeline (default when installed; cleaner OCR than EasyOCR for many PDFs)
#   Without docling package, pdf_extract falls back to EasyOCR — see requirements-docling.txt
OCR_ENGINE = os.getenv("OCR_ENGINE", "docling").lower()

# Higher DPI helps thin rules and small type in tables (GPU recommended)
OCR_DPI = int(os.getenv("OCR_DPI", "300"))

# --- Graph RAG (Faiss + Neo4j; built via graph-db/scripts/vector_index.py) ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_GRAPH_RAG_INDEX = _PROJECT_ROOT / "graph-db" / "vector-index"
GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "0").lower() in ("1", "true", "yes")
GRAPH_RAG_INDEX_DIR = Path(os.getenv("GRAPH_RAG_INDEX_DIR", str(_DEFAULT_GRAPH_RAG_INDEX)))
GRAPH_RAG_VECTOR_K = int(os.getenv("GRAPH_RAG_VECTOR_K", "12"))
GRAPH_RAG_MAX_CONTEXT_CHARS = int(os.getenv("GRAPH_RAG_MAX_CONTEXT_CHARS", "20000"))
# Hybrid Graph RAG: BM25 (lexical) + 2× dense (entity-focused vs full) fused by RRF — improves recall vs pure cosine.
GRAPH_RAG_HYBRID = os.getenv("GRAPH_RAG_HYBRID", "1").lower() in ("1", "true", "yes")
GRAPH_RAG_RRF_K = int(os.getenv("GRAPH_RAG_RRF_K", "60"))
GRAPH_RAG_DENSE_BRANCH_K = int(os.getenv("GRAPH_RAG_DENSE_BRANCH_K", "48"))
GRAPH_RAG_BM25_BRANCH_K = int(os.getenv("GRAPH_RAG_BM25_BRANCH_K", "48"))
GRAPH_RAG_DOC_SNIPPET_CHARS = int(os.getenv("GRAPH_RAG_DOC_SNIPPET_CHARS", "1200"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687").rstrip("/")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()

# --- Agentic workflow (LangChain + multi-step agents) ---
AGENT1_MODEL = os.getenv("AGENT1_MODEL", "").strip() or None  # defaults to OLLAMA_MODEL
AGENT2_MODEL = os.getenv("AGENT2_MODEL", "").strip() or None
AGENT_OCR_CHUNK_K = int(os.getenv("AGENT_OCR_CHUNK_K", "6"))
# If 1, Agent 1 uses a short Ollama call per entity for brief_summary; else heuristic text only
AGENT_LLM_SUMMARY = os.getenv("AGENT_LLM_SUMMARY", "0").lower() in ("1", "true", "yes")
