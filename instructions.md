# Runbook: GPU machine setup and UI workflow

End-to-end steps to pull the latest code, install dependencies, load Neo4j, build the Faiss vector index, configure Graph RAG, run the backend and frontend, and test the Regex Pattern Lab UI (including Graph RAG).

For NVIDIA GPU, PyTorch CUDA, and Docling cache paths, see **`GPU_SETUP.md`** in the repo root.

---

## Prerequisites (once per machine)

| Requirement | Notes |
|-------------|--------|
| **Git** | To clone/pull the repo |
| **Python 3.10+** (3.11 recommended) | For backend and `graph-db` scripts |
| **Node.js 20+** | For the Vite frontend |
| **Neo4j** | [Neo4j Desktop](https://neo4j.com/download/) or server; create a database and set a password |
| **Ollama** | [ollama.com](https://ollama.com/) for local LLMs |

Use a **short project path** (e.g. `C:\Users\<you>\Projects\ml_pattern`). Very long paths or synced folders (e.g. OneDrive) can cause issues with venvs and large caches.

---

## 1. Pull latest code

```powershell
cd C:\path\to\ml_pattern
git pull
```

---

## 2. Python virtual environment

From the **repository root**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## 3. Install Python dependencies

### Backend (FastAPI, Graph RAG: neo4j, faiss-cpu, sentence-transformers)

```powershell
pip install -r backend\requirements.txt
```

### GPU / PyTorch (optional, for OCR Docling or EasyOCR)

Install the **CUDA** build of PyTorch from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) (Windows + Pip + CUDA). Example:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If `backend\requirements.txt` installs a CPU-only torch, reinstall the CUDA wheel and recheck.

### Docling OCR pipeline (optional)

```powershell
pip install -r backend\requirements-docling.txt
```

Large Hugging Face / Docling caches may need **many GB**; you can point `HF_HOME` to a large disk (see `GPU_SETUP.md`).

### Graph DB tools (Faiss index build)

```powershell
pip install -r graph-db\requirements.txt
```

---

## 4. Neo4j: constraints and data load

1. Start the database in **Neo4j Desktop** (or your server).
2. Copy JSON files into the DBMS **import** directory (Neo4j Desktop: **Manage → Open folder → Import**).
3. In **Neo4j Browser**, run the Cypher scripts **in order**:

   - `graph-db/cypher/00_constraints.cypher`
   - `graph-db/cypher/01_load_entities.cypher`
   - `graph-db/cypher/02_load_patterns.cypher`
   - `graph-db/cypher/03_load_rules.cypher`
   - `graph-db/cypher/04_load_templates.cypher`

   Adjust `file:///…` paths in each script to match your filenames (e.g. `entities.json`). Root of each JSON file must be an **array** `[{...}, ...]`.

**Alternative:** use the Python loader:

```powershell
python graph-db\scripts\load_json_to_neo4j.py --entities path\to\entities.json --patterns path\to\patterns.json --rules path\to\rules.json --templates path\to\templates.json
```

Details: **`graph-db/README.md`**.

---

## 5. Build the Faiss vector index

Requires Neo4j populated (same machine must reach Neo4j for the script’s `MATCH` queries).

```powershell
$env:NEO4J_PASSWORD = "your-neo4j-password"
python graph-db\scripts\vector_index.py build --out graph-db\vector-index
```

This creates `graph-db/vector-index/` (`vectors.faiss`, `metadata.jsonl`, `config.json`). The first run downloads the default embedding model (sentence-transformers) and may take several minutes.

**Graph RAG retrieval (backend):** by default **`GRAPH_RAG_HYBRID=1`** uses **BM25** (lexical) over each row’s embed text plus **two dense** Faiss queries (entity/hints-only vs full query with a short doc snippet), merged by **Reciprocal Rank Fusion** (RRF). Set **`GRAPH_RAG_HYBRID=0`** for legacy single-query cosine only. Tune `GRAPH_RAG_DOC_SNIPPET_CHARS`, `GRAPH_RAG_DENSE_BRANCH_K`, `GRAPH_RAG_BM25_BRANCH_K`, `GRAPH_RAG_RRF_K` in `backend/.env`.

Smoke-test search:

```powershell
python graph-db\scripts\vector_index.py search --out graph-db\vector-index --q "your test query" -k 5
```

---

## 6. Backend environment (`backend/.env`)

Copy `backend/env.example` to `backend/.env` and set at least:

| Variable | Purpose |
|----------|--------|
| `OLLAMA_BASE_URL` | Default `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | e.g. `llama3.1:8b` (must be pulled in Ollama) |
| `OLLAMA_REFINEMENT_MODEL` | e.g. `qwen2.5:7b`; set empty to disable refinement |
| `GRAPH_RAG_ENABLED` | `1` to enable KB retrieval in the API |
| `NEO4J_URI` | e.g. `bolt://127.0.0.1:7687` |
| `NEO4J_USER` | Usually `neo4j` |
| `NEO4J_PASSWORD` | Your Neo4j password (enables **Neo4j expansion** of vector hits; if omitted, retrieval still uses Faiss + embed text from metadata) |
| `GRAPH_RAG_INDEX_DIR` | Optional absolute path to the folder containing `vectors.faiss` and `config.json`; default resolves to `graph-db/vector-index` relative to the project |

Other useful vars: `LLM_MAX_CHARS`, `LLM_NUM_CTX`, `GRAPH_RAG_VECTOR_K`, `GRAPH_RAG_MAX_CONTEXT_CHARS` (see `backend/env.example`).

---

## 7. Ollama models

```powershell
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

Use names that match `OLLAMA_MODEL` and `OLLAMA_REFINEMENT_MODEL` in `backend/.env`.

---

## 8. Frontend install

```powershell
cd frontend
npm install
```

Optional `frontend/.env` — only if the API is not on the default in `vite.config.ts`:

```env
VITE_BACKEND_ORIGIN=http://127.0.0.1:8001
```

The dev server proxies `/api` to that origin (default **8001**).

---

## 9. Run services (typical layout)

Use separate terminals or processes.

### Terminal A — Neo4j

Start the database from Neo4j Desktop (or ensure the Neo4j service is running).

### Terminal B — Ollama

Ensure the Ollama application or `ollama serve` is running.

### Terminal C — Backend API

From the **repo root**, with the venv activated:

```powershell
cd C:\path\to\ml_pattern
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8001
```

Or run `start-backend.bat` from the repo root (same port).

Checks:

```powershell
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/api/graph-rag/status
```

With Graph RAG configured you want `index_ready: true` and `enabled_flag: true` when `GRAPH_RAG_ENABLED=1`.

### Terminal D — Frontend

```powershell
cd frontend
npm run dev
```

Open the URL shown (usually **http://localhost:5173**).

---

## 10. Test the workflow in the UI

1. Open **http://localhost:5173** (or the URL Vite prints).
2. Upload a **PDF** and wait for text extraction.
3. Under **Entities**, configure at least one entity (name required; kind/hints/examples optional).
4. In **3. Model**, optionally enable **Graph RAG (Faiss + Neo4j)**.  
   - If a warning appears about the index or env, fix `GRAPH_RAG_ENABLED`, rebuild the index, or set `GRAPH_RAG_INDEX_DIR`, then refresh.
5. Click **Generate patterns** and wait (local LLMs can take several minutes on long documents).
6. In **4. Output**, verify patterns, **Graph RAG: on** (when enabled), the collapsible **hits** list, and any **Graph RAG note** if retrieval failed.

Optional API test without running the full LLM:

```powershell
curl -X POST http://127.0.0.1:8001/api/graph-rag/preview -H "Content-Type: application/json" -d "{\"q\":\"payment balance chase\",\"k\":5}"
```

---

## 11. Agentic workflow (OCR JSON → LangChain agents)

Requires the same backend deps as Graph RAG (`pip install -r backend/requirements.txt` includes **LangChain** + **langchain-ollama**).

1. Configure **`GRAPH_RAG_ENABLED=1`**, **`NEO4J_PASSWORD`**, and build the **Faiss** index (`graph-db/scripts/vector_index.py build`) so Agent 1 can query the KB.
2. In the UI, open **1b. Agentic workflow (OCR JSON)**.
3. **Upload `ocr.json`** (Azure-style: pages with `words`, `line`, `text`, etc.) — you receive a **`job_id`** (session).
4. Define **entities** in section 2 as usual (names required). Optionally add **Annotate** examples (landmark / label / value) for Agent 2.
5. **Agent 1 — Discover KB**: chunks OCR lines, builds a **session FAISS** index, runs **similarity search** per entity over OCR + searches the **global KB** vector index (and Neo4j expansion when configured). Review the summary and KB matches in the UI.
6. Adjust entities/examples, then **Agent 2 — Synthesize JSON**: uses **LangChain `ChatOllama`** to emit **`patterns` / `rules` / `templates`** arrays plus **`rationale`** (Pydantic-validated).

API routes: `POST /api/agent/ocr-upload`, `POST /api/agent/discover`, `POST /api/agent/synthesize`.

Optional: set **`AGENT_LLM_SUMMARY=1`** for a short Ollama summary per entity in Agent 1; **`AGENT1_MODEL`** / **`AGENT2_MODEL`** override the default Ollama model for each stage (`backend/env.example`).

---

## Troubleshooting

| Symptom | What to check |
|--------|----------------|
| `index_ready: false` in `/api/graph-rag/status` | Run `vector_index.py build`; set `GRAPH_RAG_INDEX_DIR` to the folder that contains `vectors.faiss` and `config.json`. |
| Graph RAG error in API/UI | Neo4j running; `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`; firewall on Bolt port. |
| `Model error` / cannot reach Ollama | Ollama running; model pulled (`ollama list`); `OLLAMA_BASE_URL`. |
| Frontend cannot reach API | Uvicorn on 8001; `VITE_BACKEND_ORIGIN` if not using default proxy. |
| OCR or GPU issues | `GPU_SETUP.md`, `OCR_ENGINE` in upload request / env, Docling vs EasyOCR installs. |
| Cypher load errors | APOC plugin installed in Neo4j; JSON arrays; `spacyPattern` stored as `spacyPatternJson` in patterns (see `graph-db/cypher/02_load_patterns.cypher`). |
| Agent 1 discover fails | `GRAPH_RAG_ENABLED` + index path; OCR JSON must parse (`normalize_azure` pages/words). |
| Agent 2 JSON error | Ollama model returns non-JSON — try `AGENT2_MODEL=qwen2.5:7b`; shorten OCR in env (`LLM_MAX_CHARS`). |

---

## Related docs

- **`graph-db/README.md`** — Neo4j model and loaders  
- **`GPU_SETUP.md`** — NVIDIA drivers, PyTorch CUDA, Docling, caches  
- **`backend/env.example`** — all backend environment variables  
