const BASE = ''

const DEFAULT_LLM_TIMEOUT_MS = 920_000 // slightly above backend Ollama timeout

function extractApiErrorMessage(status: number, bodyText: string): string {
  const trimmed = bodyText.trim()
  const fallback = trimmed || `HTTP ${status}`
  try {
    const j = JSON.parse(trimmed) as { detail?: unknown }
    if (j.detail == null) return fallback
    if (typeof j.detail === 'string') {
      const d = j.detail.trim()
      if (!d) return fallback
      // Backend bug or invisible chars: "Model error:" with no visible explanation
      if (/^model error:\s*$/i.test(d)) {
        return `${d} (empty detail) — HTTP ${status}. Full body: ${trimmed.slice(0, 500)}`
      }
      return d
    }
    return JSON.stringify(j.detail, null, 2)
  } catch {
    return fallback
  }
}

async function fetchJson<T>(
  url: string,
  init: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const { timeoutMs = DEFAULT_LLM_TIMEOUT_MS, ...rest } = init
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const r = await fetch(url, { ...rest, signal: ctrl.signal })
    if (!r.ok) {
      const t = await r.text()
      throw new Error(extractApiErrorMessage(r.status, t))
    }
    return r.json() as Promise<T>
  } catch (e: unknown) {
    const aborted =
      (e instanceof Error && e.name === 'AbortError') ||
      (typeof e === 'object' && e !== null && (e as { name?: string }).name === 'AbortError')
    if (aborted) {
      throw new Error(
        `Request timed out after ${Math.round(timeoutMs / 60000)} min. Ollama may be slow on CPU with long text — try model "gemma3:1b" or "phi3:mini", or set LLM_MAX_CHARS lower in backend .env.`,
      )
    }
    throw e
  } finally {
    clearTimeout(timer)
  }
}

export type EntitySpec = {
  name: string
  kind: string
  hints: string
}

export type UploadResponse = {
  upload_id: string
  filename: string
  pages: number
  text_preview: string
  full_text: string
  extraction_method: string
  extraction_mode?: string
  ocr_engine?: string
  ocr_dpi?: number
}

export type OcrBox = {
  id: string
  text: string
  page: number
  x0: number
  y0: number
  x1: number
  y1: number
  conf: number
}

export type OcrBoxesResponse = {
  upload_id: string
  page: number
  dpi: number
  width: number
  height: number
  image_base64: string
  boxes: OcrBox[]
}

export async function getOcrBoxes(params: {
  upload_id: string
  page?: number
  dpi?: number
}): Promise<OcrBoxesResponse> {
  const q = new URLSearchParams()
  q.set('upload_id', params.upload_id)
  q.set('page', String(params.page ?? 1))
  q.set('dpi', String(params.dpi ?? 200))
  return fetchJson(`${BASE}/api/ocr-boxes?${q.toString()}`, { method: 'GET', headers: {} })
}

export async function uploadPdf(
  file: File,
  opts?: { extractionMode?: string; ocrEngine?: string; ocrDpi?: number },
): Promise<UploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  if (opts?.extractionMode) fd.append('extraction_mode', opts.extractionMode)
  if (opts?.ocrEngine) fd.append('ocr_engine', opts.ocrEngine)
  if (opts?.ocrDpi != null) fd.append('ocr_dpi', String(opts.ocrDpi))
  const r = await fetch(`${BASE}/api/upload`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export type RegexPatternItem = {
  entity: string
  pattern: string
  flags: string
  rationale: string
  confidence_notes: string
}

export type RegexGenerateResponse = {
  patterns: RegexPatternItem[]
  raw_model_text: string
  ollama_model: string
}

export async function generateRegex(body: {
  full_text: string
  entities: EntitySpec[]
  model: string | null
}): Promise<RegexGenerateResponse> {
  return fetchJson(`${BASE}/api/generate-regex`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function listModels(): Promise<{ models: string[]; error?: string }> {
  const r = await fetch(`${BASE}/api/models`)
  if (!r.ok) return { models: [] }
  return r.json()
}

/** Same check as model dropdown: backend → Ollama /api/tags (avoids a separate route). */
export async function ollamaHealth(): Promise<{ ok: boolean; error?: string }> {
  const r = await fetch(`${BASE}/api/models`)
  if (!r.ok) {
    const t = await r.text()
    return { ok: false, error: t || `HTTP ${r.status}` }
  }
  const data = (await r.json()) as { models?: string[]; error?: string }
  if (data.error) {
    return { ok: false, error: data.error }
  }
  return { ok: true }
}

