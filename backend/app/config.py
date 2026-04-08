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
