# GPU setup: AI PatGen (Windows & Linux)

This guide covers **NVIDIA GPU** hosts for PyTorch (Docling, EasyOCR), **Ollama** (regex LLM), and the Vite frontend.

---

## CUDA 13 vs PyTorch (read this first)

- **`nvidia-smi`** shows a **CUDA Version** (e.g. 13.x). That is the **maximum** CUDA your **driver** supports — not what PyTorch must match exactly.
- **PyTorch** GPU wheels bundle their **own** CUDA runtime (commonly **12.4** `cu124` today). You install that wheel; you do **not** need to remove **CUDA 13** from your machine.
- If **no official PyTorch `cu13` wheel** exists yet for Windows, use the latest **`cu124`** (or whatever [pytorch.org/get-started](https://pytorch.org/get-started/locally/) lists for **Windows + Pip + CUDA**). It still uses your **RTX Ada** GPU as long as the driver is recent enough.

---

# A. Windows (e.g. RTX 4000 Ada, CUDA 13 driver)

### A1. Prerequisites

1. **Latest NVIDIA driver** for Ada (GeForce / RTX / workstation) from [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx).
2. **Python 3.11+** — [python.org](https://www.python.org/downloads/) — check **“Add python.exe to PATH”**.
3. **Node.js 20 LTS** — [nodejs.org](https://nodejs.org/).
4. **Git** (optional) — [git-scm.com](https://git-scm.com/download/win).

Verify GPU:

```powershell
nvidia-smi
```

You should see your GPU (e.g. **NVIDIA RTX 4000 Ada**) and a CUDA version line.

### A2. Project folder

Put the repo somewhere short, e.g. `C:\Users\<you>\Projects\ml_pattern` (avoid very long paths for pip).

```powershell
cd C:\Users\<you>\Projects\ml_pattern
```

### A3. Virtual environment (PowerShell)

```powershell
py -3.11 -m venv .venv
# If ExecutionPolicy blocks activate:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
```

### A4. PyTorch with CUDA (GPU)

Pick the **exact** line from [PyTorch Get Started](https://pytorch.org/get-started/locally/) — **Windows**, **Pip**, **CUDA 12.4** (or newest CUDA build offered). Example:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### A5. Backend dependencies + Docling

```powershell
pip install -r backend\requirements.txt
pip install -r backend\requirements-docling.txt
```

If `requirements.txt` **downgrades** `torch` to CPU-only, reinstall the **CUDA** wheel again (same `pip install torch ... --index-url ...` as A4), then recheck `torch.cuda.is_available()`.

**Disk space:** Docling + Hugging Face cache needs **many GB**. Put cache on a large drive (optional):

```powershell
mkdir D:\hf-cache   # example second drive
setx HF_HOME D:\hf-cache
setx TRANSFORMERS_CACHE D:\hf-cache
```

Open a **new** PowerShell after `setx` so env vars apply.

### A6. Backend `.env`

```powershell
copy backend\env.example backend\.env
notepad backend\.env
```

Suggested:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
OCR_USE_GPU=1
OCR_ENGINE=docling
EXTRACTION_MODE=scan
OLLAMA_STRUCTURED_JSON=1
```

### A7. Ollama (Windows, uses NVIDIA GPU when driver is OK)

1. Install from [ollama.com/download](https://ollama.com/download) (Windows).
2. Start the Ollama app / service.
3. Pull a model:

```powershell
ollama pull llama3.1:8b
ollama list
```

### A8. Frontend

```powershell
cd frontend
npm install
```

Create `frontend\.env`:

```env
VITE_BACKEND_ORIGIN=http://127.0.0.1:8001
```

### A9. Run (two terminals, venv active in the API terminal)

**Terminal 1 — API**

```powershell
cd C:\Users\<you>\Projects\ml_pattern
.\.venv\Scripts\Activate.ps1
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

**Terminal 2 — UI**

```powershell
cd C:\Users\<you>\Projects\ml_pattern\frontend
npm run dev
```

Browser: `http://localhost:5173` (Vite proxies `/api` to port **8001** by default in `vite.config.ts`).

### A10. Docling quick test (optional)

```powershell
cd C:\Users\<you>\Projects\ml_pattern
.\.venv\Scripts\Activate.ps1
python -c "from docling.document_converter import DocumentConverter; import torch; print('CUDA', torch.cuda.is_available()); print('OK')"
```

First PDF may download models — wait and keep disk space free.

---

# B. Linux GPU server (Ubuntu-style)

Use this if you deploy on a **Linux** cloud GPU instead.

### B1. Pack project (from old machine)

```bash
tar --exclude='ml_pattern/.venv' \
    --exclude='ml_pattern/frontend/node_modules' \
    -czvf ml_pattern.tar.gz ml_pattern/
```

### B2. Prerequisites

```bash
sudo apt update && sudo apt install -y git curl build-essential python3 python3-venv python3-pip
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
nvidia-smi
```

### B3. venv + PyTorch CUDA + app

```bash
cd ~/projects/ml_pattern
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt
pip install -r backend/requirements-docling.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Optional:

```bash
export HF_HOME=/path/to/bigdisk/huggingface
```

### B4. `.env`, Ollama, frontend

Same **env vars** as Windows (section A6). Ollama on Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
```

```bash
cd frontend && npm install
echo "VITE_BACKEND_ORIGIN=http://127.0.0.1:8001" > .env
```

### B5. Run

```bash
source .venv/bin/activate
cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

```bash
cd frontend && npm run dev -- --host 0.0.0.0
```

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| `torch.cuda.is_available()` is False on Windows | Reinstall GPU `torch` from [pytorch.org](https://pytorch.org/get-started/locally/); confirm `nvidia-smi`; no conflicting conda env. |
| CUDA 13 installed but no `cu13` wheel | **Expected** — use **`cu124`** (or latest offered); driver 13.x is backward compatible with PyTorch’s bundled CUDA 12.x. |
| Docling / pip “no space left” | Free disk; `pip cache purge`; set `HF_HOME` to a larger drive. |
| Ollama not using GPU | Update Ollama; check [Ollama GPU docs](https://github.com/ollama/ollama/blob/main/docs/gpu.md). |
| CORS | Add your frontend URL to `allow_origins` in `backend/app/main.py`. |

---

## One-liner summary (Windows, paths adjusted)

```powershell
cd C:\Users\<you>\Projects\ml_pattern
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r backend\requirements.txt
pip install -r backend\requirements-docling.txt
copy backend\env.example backend\.env
cd frontend; npm install; Set-Content .env "VITE_BACKEND_ORIGIN=http://127.0.0.1:8001"
```

Then: install **Ollama for Windows**, `ollama pull llama3.1:8b`, run **uvicorn** + **`npm run dev`**.
