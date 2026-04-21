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
  occurrence?: 'single' | 'multiple'
  hints: string
  /** Multiple rows per entity (e.g. one PDF each); optional source labels which PDF */
  examples?: Array<{ landmark?: string; label?: string; value?: string; source?: string }>
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
  /** Second-pass LLM (optional) */
  refinement_raw_model_text?: string
  refinement_model?: string
  /** Graph RAG (Faiss + Neo4j) when use_graph_rag was true */
  graph_rag_used?: boolean
  graph_rag_error?: string
  graph_rag_hits?: Array<{ score: number; kind: string; primary_id: string; vector_index?: number }>
}

export type RegexValidateResponse = {
  matches: Record<string, string[]>
  errors: Record<string, string>
}

export async function validateRegex(body: {
  full_text: string
  patterns: RegexPatternItem[]
}): Promise<RegexValidateResponse> {
  return fetchJson(`${BASE}/api/validate-regex`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function generateRegex(body: {
  full_text: string
  /** Extra OCR texts (e.g. other PDFs) so the model generalizes across layout variants */
  additional_full_texts?: string[]
  entities: EntitySpec[]
  model: string | null
  /** Second Ollama model: validate/repair first-pass patterns on primary OCR */
  refinement_model?: string | null
  /** Retrieve similar KB rows (Faiss) and expand in Neo4j — requires backend env */
  use_graph_rag?: boolean
}): Promise<RegexGenerateResponse> {
  return fetchJson(`${BASE}/api/generate-regex`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export type GraphRagStatus = {
  enabled_flag: boolean
  index_dir: string
  index_ready: boolean
  neo4j_configured: boolean
  neo4j_uri: string
  hybrid_enabled?: boolean
  hybrid_rrf_k?: number
  hybrid_dense_branch_k?: number
  hybrid_bm25_branch_k?: number
  doc_snippet_chars?: number
}

export async function getGraphRagStatus(): Promise<GraphRagStatus> {
  return fetchJson(`${BASE}/api/graph-rag/status`, { method: 'GET', headers: {} })
}

// --- Agentic workflow (OCR JSON → discover → synthesize) ---

export type AgentOcrUploadResponse = {
  job_id: string
  source_name: string
  page_count: number
  line_count: number
  char_count: number
  text_preview: string
}

export type KbMatchBrief = {
  kind: string
  primary_id: string
  score: number
  title: string
  summary: string
}

export type OcrChunkHit = {
  chunk_id: string
  page: number
  text_excerpt: string
  relevance_note: string
}

export type EntityDiscoveryResult = {
  entity_name: string
  kind: string
  kb_matches: KbMatchBrief[]
  ocr_chunk_hits: OcrChunkHit[]
  brief_summary: string
}

export type AgentDiscoverResponse = {
  job_id: string
  ocr_chunks_indexed: number
  entities: EntityDiscoveryResult[]
  graph_rag_error: string
  notes: string
}

export type ValidatedEntityOcr = {
  name: string
  kind: string
  landmark: string
  label: string
  value: string
  hints: string
}

export type AgentArtifactEnvelope = {
  patterns: Record<string, unknown>[]
  rules: Record<string, unknown>[]
  templates: Record<string, unknown>[]
  rationale: string
}

export type AgentSynthesizeResponse = {
  job_id: string
  artifacts: AgentArtifactEnvelope
  raw_model_text: string
  ollama_model: string
  error: string
}

export async function uploadAgentOcr(file: File): Promise<AgentOcrUploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  return fetchJson(`${BASE}/api/agent/ocr-upload`, { method: 'POST', body: fd })
}

export async function agentDiscover(body: {
  job_id: string
  entities: EntitySpec[]
  kb_vector_k?: number
  ocr_chunk_k?: number
}): Promise<AgentDiscoverResponse> {
  return fetchJson(`${BASE}/api/agent/discover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function agentSynthesize(body: {
  job_id: string
  validated: ValidatedEntityOcr[]
  model?: string | null
  extra_instructions?: string
}): Promise<AgentSynthesizeResponse> {
  return fetchJson(`${BASE}/api/agent/synthesize`, {
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

