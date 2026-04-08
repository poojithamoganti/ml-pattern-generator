import { useCallback, useEffect, useState } from 'react'
import {
  generateRegex,
  getOcrBoxes,
  listModels,
  ollamaHealth,
  uploadPdf,
  validateRegex,
  type EntitySpec,
  type OcrBox,
  type OcrBoxesResponse,
  type RegexGenerateResponse,
  type RegexPatternItem,
  type UploadResponse,
} from './api'
import './App.css'

/** Default when backend env OLLAMA_MODEL is unset; must match a pulled model name. */
const DEFAULT_OLLAMA_MODEL = 'llama3.1:8b'

const ENTITY_KINDS = [
  { value: 'text', label: 'Text' },
  { value: 'date', label: 'Date' },
  { value: 'amount', label: 'Amount' },
  { value: 'currency', label: 'Currency' },
  { value: 'number', label: 'Number' },
  { value: 'email', label: 'Email' },
  { value: 'phone', label: 'Phone' },
  { value: 'address', label: 'Address' },
  { value: 'id', label: 'ID / reference' },
  { value: 'other', label: 'Other' },
] as const

/** Extra layout PDFs: text for generation + server id for bbox annotation */
type ExtraSample = { id: string; filename: string; full_text: string; upload_id: string; pages: number }

const ANNOTATE_PRIMARY = '__primary__'

type SavedPickExample = {
  /** Which PDF these picks were made on (shown in API as source) */
  sourceLabel: string
  landmark?: { text: string; box_id: string; page: number }
  label?: { text: string; box_id: string; page: number }
  value?: { text: string; box_id: string; page: number }
}

type EntityForm = {
  id: string
  name: string
  kind: string
  occurrence: 'single' | 'multiple'
  hints: string
  picked_landmark?: { text: string; box_id: string; page: number }
  picked_label?: { text: string; box_id: string; page: number }
  picked_value?: { text: string; box_id: string; page: number }
  /** PDF label at time of last box pick (for draft + API source) */
  pickSourceLabel?: string
  /** Saved landmark/label/value groups — use one row per entity name; add multiple for different PDFs */
  examples?: SavedPickExample[]
}

type ApiExample = { landmark?: string; label?: string; value?: string; source?: string }

function exampleKey(e: ApiExample): string {
  return [e.source || '', e.landmark || '', e.label || '', e.value || ''].join('\x01')
}

function docLabelForAnnotate(
  annotateDocId: string,
  upload: UploadResponse | null,
  extraSamples: ExtraSample[],
): string {
  if (!upload) return 'document'
  if (annotateDocId === ANNOTATE_PRIMARY) return `Primary — ${upload.filename}`
  const s = extraSamples.find((x) => x.id === annotateDocId)
  return s ? `Sample — ${s.filename}` : `Primary — ${upload.filename}`
}

function picksToApiDraft(row: EntityForm): ApiExample | null {
  if (!row.picked_landmark && !row.picked_label && !row.picked_value) return null
  return {
    landmark: row.picked_landmark?.text,
    label: row.picked_label?.text,
    value: row.picked_value?.text,
    source: row.pickSourceLabel,
  }
}

/** Merge rows with the same entity name; combine saved examples + current picks into one EntitySpec. */
function toSpecs(rows: EntityForm[]): EntitySpec[] {
  const named = rows.filter((r) => r.name.trim())
  const groups = new Map<string, EntityForm[]>()
  for (const r of named) {
    const k = r.name.trim().toLowerCase()
    if (!groups.has(k)) groups.set(k, [])
    groups.get(k)!.push(r)
  }
  const out: EntitySpec[] = []
  for (const [, group] of groups) {
    const head = group[0]
    const merged: ApiExample[] = []
    const seen = new Set<string>()
    const pushEx = (e: ApiExample) => {
      if (!(e.landmark || e.label || e.value)) return
      const k = exampleKey(e)
      if (seen.has(k)) return
      seen.add(k)
      merged.push(e)
    }
    const hintParts: string[] = []
    for (const r of group) {
      for (const ex of r.examples || []) {
        pushEx({
          landmark: ex.landmark?.text,
          label: ex.label?.text,
          value: ex.value?.text,
          source: ex.sourceLabel,
        })
      }
      const draft = picksToApiDraft(r)
      if (draft) pushEx(draft)
      if (r.hints.trim()) hintParts.push(r.hints.trim())
    }
    const examples: Record<string, string>[] = merged.map((e) => {
      const o: Record<string, string> = {}
      if (e.landmark != null && e.landmark !== '') o.landmark = e.landmark
      if (e.label != null && e.label !== '') o.label = e.label
      if (e.value != null && e.value !== '') o.value = e.value
      if (e.source != null && e.source !== '') o.source = e.source
      return o
    })
    out.push({
      name: head.name.trim(),
      kind: head.kind || 'text',
      occurrence: head.occurrence || 'single',
      hints: hintParts.join('\n\n---\n'),
      examples,
    })
  }
  return out
}

/** Pretty-print if the model returned JSON; otherwise show verbatim. */
function prettyModelRaw(raw: string): string {
  const t = raw.trim()
  if (!t) return '(empty)'
  let slice = t
  const fence = /^```(?:json)?\s*([\s\S]*?)```/m.exec(t)
  if (fence) slice = fence[1].trim()
  const brace = slice.indexOf('{')
  if (brace !== -1) {
    const end = slice.lastIndexOf('}')
    if (end > brace) slice = slice.slice(brace, end + 1)
  }
  try {
    return JSON.stringify(JSON.parse(slice), null, 2)
  } catch {
    return raw
  }
}

function normalizeForMatch(s: string): string {
  return (s || '')
    .toLowerCase()
    .replace(/[\s\r\n\t]+/g, ' ')
    .replace(/[^a-z0-9.,$/-]/g, '')
    .trim()
}

function resolveAnnotateCtx(
  u: UploadResponse | null,
  extras: ExtraSample[],
  docId: string,
): { upload_id: string; pages: number; filename: string } | null {
  if (!u) return null
  if (docId === ANNOTATE_PRIMARY) {
    return { upload_id: u.upload_id, pages: u.pages, filename: u.filename }
  }
  const ex = extras.find((s) => s.id === docId)
  if (!ex) {
    return { upload_id: u.upload_id, pages: u.pages, filename: u.filename }
  }
  return { upload_id: ex.upload_id, pages: ex.pages, filename: ex.filename }
}

export default function App() {
  const [upload, setUpload] = useState<UploadResponse | null>(null)
  /** Extra PDFs sent with generate; each keeps upload_id for optional bbox annotation */
  const [extraSamples, setExtraSamples] = useState<ExtraSample[]>([])
  /** Which uploaded PDF to show in Annotate mode: primary or one extra sample id */
  const [annotateDocId, setAnnotateDocId] = useState<string>(ANNOTATE_PRIMARY)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [entities, setEntities] = useState<EntityForm[]>([
    { id: '1', name: 'Invoice Number', kind: 'id', occurrence: 'single', hints: '', examples: [] },
  ])
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState(DEFAULT_OLLAMA_MODEL)
  const [result, setResult] = useState<RegexGenerateResponse | null>(null)
  const [validationUpload, setValidationUpload] = useState<UploadResponse | null>(null)
  const [validation, setValidation] = useState<{ matches: Record<string, string[]>; errors: Record<string, string> } | null>(null)
  const [validationPage, setValidationPage] = useState(1)
  const [validationBoxesDpi, setValidationBoxesDpi] = useState(200)
  const [validationOcrView, setValidationOcrView] = useState<OcrBoxesResponse | null>(null)
  const [showValidationHighlights, setShowValidationHighlights] = useState(true)
  const [highlightBoxIds, setHighlightBoxIds] = useState<Set<string>>(new Set())
  const [copiedEntity, setCopiedEntity] = useState<string | null>(null)
  const [rawPanelOpen, setRawPanelOpen] = useState(true)
  const [extractionMode, setExtractionMode] = useState<'scan' | 'auto' | 'embedded'>('scan')
  const [ocrEngine, setOcrEngine] = useState<'paddle' | 'easyocr' | 'docling'>('easyocr')
  const [ocrDpi, setOcrDpi] = useState(300)
  const [docMode, setDocMode] = useState<'text' | 'annotate'>('text')
  const [ocrPage, setOcrPage] = useState(1)
  const [ocrDpiAnnotate, setOcrDpiAnnotate] = useState(200)
  const [ocrView, setOcrView] = useState<OcrBoxesResponse | null>(null)
  const [activePick, setActivePick] = useState<{ entityId: string; field: 'landmark' | 'label' | 'value' } | null>(
    null,
  )
  const [busyHint, setBusyHint] = useState('')
  const [elapsedSec, setElapsedSec] = useState(0)
  const [ollamaWarn, setOllamaWarn] = useState<string | null>(null)

  const loadOcrPage = useCallback(
    async (page: number, docIdOverride?: string, dpiOverride?: number) => {
      const docId = docIdOverride ?? annotateDocId
      const ctx = resolveAnnotateCtx(upload, extraSamples, docId)
      if (!ctx) return
      const dpi = dpiOverride ?? ocrDpiAnnotate
      setErr(null)
      setBusy(true)
      setBusyHint('Loading OCR boxes…')
      try {
        const v = await getOcrBoxes({ upload_id: ctx.upload_id, page, dpi })
        setOcrPage(page)
        setOcrView(v)
      } catch (e: unknown) {
        setErr(String(e))
      } finally {
        setBusy(false)
        setBusyHint('')
      }
    },
    [upload, extraSamples, annotateDocId, ocrDpiAnnotate],
  )

  useEffect(() => {
    if (annotateDocId === ANNOTATE_PRIMARY) return
    if (!extraSamples.some((s) => s.id === annotateDocId)) {
      setAnnotateDocId(ANNOTATE_PRIMARY)
      if (docMode === 'annotate' && upload) {
        void loadOcrPage(1, ANNOTATE_PRIMARY)
      }
    }
  }, [extraSamples, annotateDocId, docMode, upload, loadOcrPage])

  useEffect(() => {
    if (!busy) {
      setElapsedSec(0)
      return
    }
    setElapsedSec(0)
    const id = window.setInterval(() => setElapsedSec((s) => s + 1), 1000)
    return () => window.clearInterval(id)
  }, [busy])

  useEffect(() => {
    ollamaHealth().then((h) => {
      if (!h.ok) {
        setOllamaWarn(
          h.error
            ? `Cannot reach Ollama via the API: ${h.error}. Is the backend URL correct (see frontend .env VITE_BACKEND_ORIGIN) and is Ollama running?`
            : 'Could not verify Ollama. Start the Ollama app, then refresh.',
        )
      } else {
        setOllamaWarn(null)
      }
    })
  }, [])

  useEffect(() => {
    listModels().then((d) => {
      const list = d.models || []
      setModels(list)
      setModel((prev) => {
        if (list.includes(prev)) return prev
        if (list.includes(DEFAULT_OLLAMA_MODEL)) return DEFAULT_OLLAMA_MODEL
        return list[0] ?? prev
      })
    })
  }, [])

  const onFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    setErr(null)
    setBusy(true)
    setBusyHint('Uploading PDF and extracting text…')
    setResult(null)
    try {
      const u = await uploadPdf(f, {
        extractionMode: extractionMode,
        ocrEngine: ocrEngine,
        ocrDpi: ocrDpi,
      })
      setUpload(u)
      setAnnotateDocId(ANNOTATE_PRIMARY)
      setOcrView(null)
      setActivePick(null)
      if (docMode === 'annotate') {
        setBusyHint('Preparing OCR boxes for annotation…')
        const v = await getOcrBoxes({ upload_id: u.upload_id, page: 1, dpi: ocrDpiAnnotate })
        setOcrPage(1)
        setOcrView(v)
      }
    } catch (er: unknown) {
      setErr(String(er))
    } finally {
      setBusy(false)
      setBusyHint('')
    }
  }, [extractionMode, ocrEngine, ocrDpi, docMode, ocrDpiAnnotate])

  const onExtraSample = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0]
      if (!f) return
      if (extraSamples.length >= 12) {
        setErr('You can add at most 12 extra sample PDFs.')
        e.target.value = ''
        return
      }
      setErr(null)
      setBusy(true)
      setBusyHint('Uploading extra sample PDF…')
      try {
        const u = await uploadPdf(f, {
          extractionMode: extractionMode,
          ocrEngine: ocrEngine,
          ocrDpi: ocrDpi,
        })
        setExtraSamples((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            filename: u.filename,
            full_text: u.full_text,
            upload_id: u.upload_id,
            pages: u.pages,
          },
        ])
      } catch (er: unknown) {
        setErr(String(er))
      } finally {
        setBusy(false)
        setBusyHint('')
        e.target.value = ''
      }
    },
    [extraSamples.length, extractionMode, ocrEngine, ocrDpi],
  )

  const removeExtraSample = useCallback(
    (id: string) => {
      setExtraSamples((prev) => prev.filter((s) => s.id !== id))
      if (id === annotateDocId) {
        setAnnotateDocId(ANNOTATE_PRIMARY)
        if (docMode === 'annotate' && upload) {
          void loadOcrPage(1, ANNOTATE_PRIMARY)
        }
      }
    },
    [annotateDocId, docMode, upload, loadOcrPage],
  )

  const onValidationFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    setErr(null)
    setBusy(true)
    setBusyHint('Uploading validation PDF and extracting text…')
    setValidation(null)
    setValidationOcrView(null)
    setHighlightBoxIds(new Set())
    try {
      const u = await uploadPdf(f, {
        extractionMode: extractionMode,
        ocrEngine: ocrEngine,
        ocrDpi: ocrDpi,
      })
      setValidationUpload(u)
      setValidationPage(1)
    } catch (er: unknown) {
      setErr(String(er))
    } finally {
      setBusy(false)
      setBusyHint('')
    }
  }, [extractionMode, ocrEngine, ocrDpi])

  const loadValidationOcrPage = async (page: number) => {
    if (!validationUpload) return
    setErr(null)
    setBusy(true)
    setBusyHint('Loading validation OCR boxes…')
    try {
      const v = await getOcrBoxes({ upload_id: validationUpload.upload_id, page, dpi: validationBoxesDpi })
      setValidationPage(page)
      setValidationOcrView(v)
    } catch (e: unknown) {
      setErr(String(e))
    } finally {
      setBusy(false)
      setBusyHint('')
    }
  }

  const runValidate = async () => {
    if (!validationUpload?.full_text) {
      setErr('Upload a validation PDF first.')
      return
    }
    if (!result?.patterns?.length) {
      setErr('Generate patterns first.')
      return
    }
    setErr(null)
    setBusy(true)
    setBusyHint('Validating regex patterns on the validation document…')
    try {
      const out = await validateRegex({
        full_text: validationUpload.full_text,
        patterns: result.patterns as RegexPatternItem[],
      })
      setValidation(out)
    } catch (e: unknown) {
      setErr(String(e))
    } finally {
      setBusy(false)
      setBusyHint('')
    }
  }

  useEffect(() => {
    if (!validation || !validationOcrView || !showValidationHighlights) {
      setHighlightBoxIds(new Set())
      return
    }
    const want = new Set<string>()
    const boxes = validationOcrView.boxes || []
    const normBox = boxes.map((b) => ({ id: b.id, t: normalizeForMatch(b.text) }))

    const valsAll = Object.values(validation.matches || {}).flat()
    for (const v of valsAll) {
      const nv = normalizeForMatch(v)
      if (!nv) continue
      // Prefer exact normalized match
      let hit = normBox.find((b) => b.t === nv)
      if (!hit) {
        // Containment (handles extra punctuation/spaces)
        hit = normBox.find((b) => b.t && (b.t.includes(nv) || nv.includes(b.t)))
      }
      if (hit) want.add(hit.id)
    }
    setHighlightBoxIds(want)
  }, [validation, validationOcrView, showValidationHighlights])

  const onPickBox = (box: OcrBox) => {
    if (!activePick) return
    const pickSrc = docLabelForAnnotate(annotateDocId, upload, extraSamples)
    setEntities((prev) =>
      prev.map((r) => {
        if (r.id !== activePick.entityId) return r
        if (activePick.field === 'landmark') {
          const nextHints = r.hints.includes('Landmark:')
            ? r.hints
            : `${r.hints}${r.hints.trim() ? '\n' : ''}Landmark: "${box.text}"`
          return {
            ...r,
            picked_landmark: { text: box.text, box_id: box.id, page: box.page },
            hints: nextHints,
            pickSourceLabel: pickSrc,
          }
        }
        if (activePick.field === 'label') {
          const nextHints = r.hints.includes('"')
            ? r.hints
            : `${r.hints}${r.hints.trim() ? '\n' : ''}Label text: "${box.text}"`
          return {
            ...r,
            picked_label: { text: box.text, box_id: box.id, page: box.page },
            hints: nextHints,
            pickSourceLabel: pickSrc,
          }
        }
        const nextHints =
          r.hints.trim().length === 0
            ? `Example value: "${box.text}"`
            : `${r.hints}\nExample value: "${box.text}"`
        return {
          ...r,
          picked_value: { text: box.text, box_id: box.id, page: box.page },
          hints: nextHints,
          pickSourceLabel: pickSrc,
        }
      }),
    )
    setActivePick(null)
  }

  const savePicksAsExample = (entityId: string) => {
    setEntities((prev) =>
      prev.map((r) => {
        if (r.id !== entityId) return r
        if (!r.picked_landmark && !r.picked_label && !r.picked_value) return r
        const sourceLabel = r.pickSourceLabel ?? docLabelForAnnotate(annotateDocId, upload, extraSamples)
        const nextEx: SavedPickExample = {
          sourceLabel,
          landmark: r.picked_landmark,
          label: r.picked_label,
          value: r.picked_value,
        }
        return {
          ...r,
          examples: [...(r.examples || []), nextEx],
          picked_landmark: undefined,
          picked_label: undefined,
          picked_value: undefined,
          pickSourceLabel: undefined,
        }
      }),
    )
  }

  const removeSavedExample = (entityId: string, index: number) => {
    setEntities((prev) =>
      prev.map((r) => {
        if (r.id !== entityId) return r
        const next = [...(r.examples || [])]
        next.splice(index, 1)
        return { ...r, examples: next }
      }),
    )
  }

  const addEntity = () => {
    setEntities((prev) => [
      ...prev,
      { id: crypto.randomUUID(), name: '', kind: 'text', occurrence: 'single', hints: '', examples: [] },
    ])
  }

  const updateEntity = (id: string, patch: Partial<EntityForm>) => {
    setEntities((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)))
  }

  const removeEntity = (id: string) => {
    setEntities((prev) => (prev.length <= 1 ? prev : prev.filter((r) => r.id !== id)))
  }

  const runGenerate = async () => {
    if (!upload?.full_text) {
      setErr('Upload a PDF first.')
      return
    }
    const specs = toSpecs(entities)
    if (!specs.length) {
      setErr('Add at least one entity with a name.')
      return
    }
    setErr(null)
    setBusy(true)
    setBusyHint(
      extraSamples.length
        ? `Ollama is generating from 1 primary + ${extraSamples.length} extra sample(s) — may take longer.`
        : 'Ollama is generating regex — CPU inference can take several minutes on long text. Watch the timer below.',
    )
    try {
      const gen = await generateRegex({
        full_text: upload.full_text,
        additional_full_texts: extraSamples.map((s) => s.full_text),
        entities: specs,
        model: model || null,
      })
      setResult(gen)
      setCopiedEntity(null)
      setRawPanelOpen(true)
    } catch (er: unknown) {
      setErr(String(er))
    } finally {
      setBusy(false)
      setBusyHint('')
    }
  }

  const copyPattern = async (entity: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedEntity(entity)
      window.setTimeout(() => {
        setCopiedEntity((c) => (c === entity ? null : c))
      }, 2000)
    } catch {
      setErr('Could not copy to clipboard.')
    }
  }

  const downloadPatternsJson = () => {
    if (!result) return
    const payload = {
      ollama_model: result.ollama_model,
      generated_at: new Date().toISOString(),
      patterns: result.patterns.map((p) => ({
        entity: p.entity,
        pattern: p.pattern,
        flags: p.flags,
        rationale: p.rationale,
        confidence_notes: p.confidence_notes,
      })),
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json;charset=utf-8',
    })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    const safe = result.ollama_model.replace(/[/\\:*?"<>|]/g, '_')
    a.download = `patterns-${safe}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const annotateCtx = upload ? resolveAnnotateCtx(upload, extraSamples, annotateDocId) : null

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <h1 className="brand">AI PatGen</h1>
      </header>

      <div className="app">
        {ollamaWarn && <div className="warn-banner">{ollamaWarn}</div>}

        {busy && busyHint && (
          <div className="busy-banner" role="status" aria-live="polite">
            <p>{busyHint}</p>
            <p className="elapsed">{elapsedSec}s elapsed</p>
          </div>
        )}

        <section className="card">
          <h2>1. Document</h2>
          <div className="extract-opts">
          <label>
            Input
            <select
              value={docMode}
              onChange={(e) => {
                const v = e.target.value as typeof docMode
                setDocMode(v)
                setOcrView(null)
                setActivePick(null)
                if (v === 'annotate' && upload?.upload_id) {
                  loadOcrPage(1)
                }
              }}
            >
              <option value="text">Extract text</option>
              <option value="annotate">Annotate</option>
            </select>
          </label>
          <label>
            Mode
            <select
              value={extractionMode}
              onChange={(e) => setExtractionMode(e.target.value as typeof extractionMode)}
            >
              <option value="scan">Scan — always OCR (recommended)</option>
              <option value="auto">Auto — text layer if present, else OCR</option>
              <option value="embedded">Embedded text only (no OCR)</option>
            </select>
          </label>
          <label>
            OCR engine
            <select
              value={ocrEngine}
              onChange={(e) => setOcrEngine(e.target.value as typeof ocrEngine)}
              disabled={extractionMode === 'embedded'}
            >
              <option value="paddle">Paddle PP-Structure (layout + tables)</option>
              <option value="easyocr">EasyOCR + layout sort</option>
              <option value="docling">Docling (layout + OCR — install backend extra)</option>
            </select>
          </label>
          {docMode === 'annotate' && (
            <label>
              Boxes DPI
              <input
                type="number"
                min={100}
                max={400}
                step={50}
                value={ocrDpiAnnotate}
                onChange={(e) => {
                  const n = Number(e.target.value) || 200
                  setOcrDpiAnnotate(n)
                  if (upload) void loadOcrPage(ocrPage, undefined, n)
                }}
                disabled={busy}
                title="Lower DPI loads faster; affects box coordinates for annotation only."
              />
            </label>
          )}
          <label>
            Render DPI
            <input
              type="number"
              min={150}
              max={600}
              step={50}
              value={ocrDpi}
              onChange={(e) => setOcrDpi(Number(e.target.value) || 300)}
              disabled={extractionMode === 'embedded' || ocrEngine === 'docling'}
              title={ocrEngine === 'docling' ? 'Docling uses its own pipeline; DPI applies to Paddle/EasyOCR only.' : undefined}
            />
          </label>
        </div>
        <label className="file">
          <input type="file" accept="application/pdf" onChange={onFile} disabled={busy} />
          <span>{upload ? upload.filename : 'Choose PDF'}</span>
        </label>
        {upload && (
          <p className="meta">
            {upload.pages} page(s) · {upload.extraction_method}
            {upload.extraction_mode != null && upload.ocr_engine != null && (
              <>
                {' '}
                · mode {upload.extraction_mode} · {upload.ocr_engine} · {upload.ocr_dpi ?? ocrDpi} DPI
              </>
            )}
            {upload.full_text.length > 0 && ` · ${upload.full_text.length} characters`}
            {extraSamples.length > 0 && ` · +${extraSamples.length} extra layout sample(s) for generation`}
          </p>
        )}
        {upload && (
          <div className="extra-sample-block">
            {extraSamples.length > 0 && (
              <ul className="extra-samples">
                {extraSamples.map((s) => (
                  <li key={s.id}>
                    <span title={`${s.full_text.length.toLocaleString()} characters`}>{s.filename}</span>
                    <button
                      type="button"
                      className="btn secondary small"
                      onClick={() => removeExtraSample(s.id)}
                      disabled={busy}
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <div className="extra-sample-row">
              <label className="file file-secondary">
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={onExtraSample}
                  disabled={busy || extraSamples.length >= 12}
                />
                <span>
                  {extraSamples.length >= 12 ? 'Maximum 12 extra samples' : 'Add sample PDF (other layout)'}
                </span>
              </label>
              {extraSamples.length > 0 && (
                <button
                  type="button"
                  className="btn secondary"
                  onClick={() => {
                    setExtraSamples([])
                    if (annotateDocId !== ANNOTATE_PRIMARY) {
                      setAnnotateDocId(ANNOTATE_PRIMARY)
                      if (docMode === 'annotate' && upload) void loadOcrPage(1, ANNOTATE_PRIMARY)
                    }
                  }}
                  disabled={busy}
                >
                  Clear extra samples
                </button>
              )}
            </div>
            <p className="muted small">
              Extra PDFs add OCR text for generation and appear below in Annotate so you can pick landmark / label / value
              boxes on each layout.
            </p>
          </div>
        )}
        {upload && (
          <details className="preview" open>
            <summary>
              View extracted text ({upload.full_text.length.toLocaleString()} characters)
            </summary>
            <div className="preview-toolbar">
              <button
                type="button"
                className="btn secondary"
                onClick={() => {
                  const blob = new Blob([upload.full_text], { type: 'text/plain;charset=utf-8' })
                  const a = document.createElement('a')
                  a.href = URL.createObjectURL(blob)
                  a.download = `${upload.filename.replace(/\.pdf$/i, '')}_extracted.txt`
                  a.click()
                  URL.revokeObjectURL(a.href)
                }}
              >
                Download as .txt
              </button>
            </div>
            <pre className="preview-body">{upload.full_text}</pre>
          </details>
        )}

        {docMode === 'annotate' && upload && annotateCtx && (
          <div className="annotate">
            <div className="annotate-toolbar">
              <label className="annotate-doc-select">
                Annotate document
                <select
                  value={annotateDocId}
                  onChange={(e) => {
                    const v = e.target.value
                    setAnnotateDocId(v)
                    void loadOcrPage(1, v)
                  }}
                  disabled={busy}
                  aria-label="Choose which PDF to annotate with bounding boxes"
                >
                  <option value={ANNOTATE_PRIMARY}>Primary — {upload.filename}</option>
                  {extraSamples.map((s) => (
                    <option key={s.id} value={s.id}>
                      Sample — {s.filename}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="btn secondary"
                onClick={() => loadOcrPage(Math.max(1, ocrPage - 1))}
                disabled={busy || ocrPage <= 1}
              >
                Prev page
              </button>
              <div className="annotate-page">
                Page {ocrPage} / {annotateCtx.pages}
                <span className="annotate-doc-name" title={annotateCtx.filename}>
                  ({annotateCtx.filename})
                </span>
              </div>
              <button
                type="button"
                className="btn secondary"
                onClick={() => loadOcrPage(Math.min(annotateCtx.pages, ocrPage + 1))}
                disabled={busy || ocrPage >= annotateCtx.pages}
              >
                Next page
              </button>
              <button
                type="button"
                className="btn secondary"
                onClick={() => loadOcrPage(ocrPage)}
                disabled={busy}
              >
                Refresh boxes
              </button>
              {activePick && <div className="pick-hint">Click a box to set {activePick.field}…</div>}
            </div>

            {ocrView && (
              <div className="ocr-canvas" style={{ aspectRatio: `${ocrView.width} / ${ocrView.height}` }}>
                <img
                  className="ocr-image"
                  src={`data:image/png;base64,${ocrView.image_base64}`}
                  alt={`OCR page ${ocrView.page}`}
                />
                {ocrView.boxes.map((b) => {
                  const left = (b.x0 / ocrView.width) * 100
                  const top = (b.y0 / ocrView.height) * 100
                  const w = ((b.x1 - b.x0) / ocrView.width) * 100
                  const h = ((b.y1 - b.y0) / ocrView.height) * 100
                  return (
                    <button
                      key={b.id}
                      type="button"
                      className="ocr-box"
                      style={{ left: `${left}%`, top: `${top}%`, width: `${w}%`, height: `${h}%` }}
                      title={b.text}
                      onClick={() => onPickBox(b)}
                      disabled={!activePick}
                    />
                  )
                })}
              </div>
            )}
          </div>
        )}
      </section>

        <section className="card">
          <h2>2. Entities</h2>
          <p className="section-lead">
            One row per logical field (e.g. <strong>New Balance</strong>). In <strong>Annotate</strong>, pick landmark /
            label / value on the <strong>primary</strong> PDF, click <strong>Save picks as example</strong>, switch{' '}
            <strong>Annotate document</strong> to a sample PDF, pick again for the <em>same</em> entity row, save again
            — or add a second row with the same name; all examples merge into <strong>one</strong> pattern at generate
            time.
          </p>
          {entities.map((row) => (
            <div key={row.id} className="entity-row">
              <input
                placeholder="Entity name"
                value={row.name}
                onChange={(e) => updateEntity(row.id, { name: e.target.value })}
                aria-label="Entity name"
              />
              <label className="entity-kind-label">
                Type
                <select
                  value={row.kind}
                  onChange={(e) => updateEntity(row.id, { kind: e.target.value })}
                  aria-label="Entity value type"
                >
                  {ENTITY_KINDS.map((k) => (
                    <option key={k.value} value={k.value}>
                      {k.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="entity-kind-label">
                Occurrence
                <select
                  value={row.occurrence}
                  onChange={(e) =>
                    updateEntity(row.id, { occurrence: e.target.value as 'single' | 'multiple' })
                  }
                  aria-label="Entity occurrence"
                >
                  <option value="single">Single on page</option>
                  <option value="multiple">Multiple on page</option>
                </select>
              </label>
              <button type="button" className="btn ghost" onClick={() => removeEntity(row.id)} title="Remove">
                ✕
              </button>
              {docMode === 'annotate' && (
                <div className="entity-annotate-block">
                  <div className="pick-row">
                    <button
                      type="button"
                      className="btn secondary"
                      onClick={() => setActivePick({ entityId: row.id, field: 'landmark' })}
                      disabled={!ocrView || busy}
                    >
                      Pick landmark
                    </button>
                    <button
                      type="button"
                      className="btn secondary"
                      onClick={() => setActivePick({ entityId: row.id, field: 'label' })}
                      disabled={!ocrView || busy}
                    >
                      Pick label
                    </button>
                    <button
                      type="button"
                      className="btn secondary"
                      onClick={() => setActivePick({ entityId: row.id, field: 'value' })}
                      disabled={!ocrView || busy}
                    >
                      Pick value
                    </button>
                    <div className="pick-summary">
                      {row.pickSourceLabel && (
                        <span className="pick-doc">
                          From: <strong>{row.pickSourceLabel}</strong>
                        </span>
                      )}
                      {row.picked_landmark?.text ? (
                        <span>
                          Landmark: <code>{row.picked_landmark.text}</code>
                        </span>
                      ) : (
                        <span>Landmark: —</span>
                      )}
                      {row.picked_label?.text ? (
                        <span>
                          Label: <code>{row.picked_label.text}</code>
                        </span>
                      ) : (
                        <span>Label: —</span>
                      )}
                      {row.picked_value?.text ? (
                        <span>
                          Value: <code>{row.picked_value.text}</code>
                        </span>
                      ) : (
                        <span>Value: —</span>
                      )}
                    </div>
                  </div>
                  <div className="pick-actions">
                    <button
                      type="button"
                      className="btn secondary"
                      onClick={() => savePicksAsExample(row.id)}
                      disabled={busy || (!row.picked_landmark && !row.picked_label && !row.picked_value)}
                      title="Store current picks as one training example; then switch PDF and pick again, or generate."
                    >
                      Save picks as example
                    </button>
                  </div>
                  {(row.examples?.length ?? 0) > 0 && (
                    <ul className="saved-examples">
                      {row.examples!.map((ex, i) => (
                        <li key={`${row.id}-ex-${i}`}>
                          <span className="saved-examples-source">{ex.sourceLabel}</span>
                          <span className="saved-examples-bits">
                            {[ex.landmark?.text && `lm: ${ex.landmark.text}`, ex.label?.text && `lb: ${ex.label.text}`, ex.value?.text && `val: ${ex.value.text}`]
                              .filter(Boolean)
                              .join(' · ')}
                          </span>
                          <button
                            type="button"
                            className="btn ghost small"
                            onClick={() => removeSavedExample(row.id, i)}
                            disabled={busy}
                            aria-label="Remove saved example"
                          >
                            Remove
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
              <label className="entity-hints-label">
                Hints (position, column, layout, format)
                <textarea
                  placeholder="e.g. top-right of first page, next to label ‘Balance’, table column 3"
                  value={row.hints}
                  onChange={(e) => updateEntity(row.id, { hints: e.target.value })}
                  rows={2}
                />
              </label>
            </div>
          ))}
        <button type="button" className="btn secondary" onClick={addEntity}>
          + Add entity
        </button>
      </section>

      <section className="card">
        <h2>3. Model</h2>
        <div className="model-row">
          <label className="model-select-label">
            Ollama model
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.length === 0 && (
                <option value={DEFAULT_OLLAMA_MODEL}>{DEFAULT_OLLAMA_MODEL} (default)</option>
              )}
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <button type="button" className="btn primary" onClick={runGenerate} disabled={busy}>
            Generate patterns
          </button>
        </div>
      </section>

      {err && (
        <div className="error">
          <p className="error-text">{err}</p>
          {err.includes('Model error') && (
            <p className="error-hint">
              Check the API terminal for the full traceback. Common fixes: Ollama running, model pulled (
              <code>ollama pull …</code>), and <code>frontend/.env</code> <code>VITE_BACKEND_ORIGIN</code> matching
              your uvicorn port (default Vite proxy is 8001).
            </p>
          )}
        </div>
      )}

      {result && (
        <section className="card">
          <div className="output-header">
            <h2>4. Output</h2>
            <div className="output-actions">
              <button type="button" className="btn secondary" onClick={downloadPatternsJson}>
                Download JSON
              </button>
            </div>
          </div>
          <p className="output-meta">Model: {result.ollama_model}</p>
          <ul className="patterns">
            {result.patterns.map((p) => (
              <li key={p.entity}>
                <h3>{p.entity}</h3>
                <div className="pattern-copy-row">
                  <code className="pattern">{p.pattern || '(empty)'}</code>
                  <button
                    type="button"
                    className="btn secondary btn-copy"
                    onClick={() => copyPattern(p.entity, p.pattern)}
                    disabled={!p.pattern.trim()}
                  >
                    {copiedEntity === p.entity ? 'Copied' : 'Copy'}
                  </button>
                </div>
                {p.flags && <div className="flags">Flags: {p.flags}</div>}
                {p.rationale && <p className="rationale">{p.rationale}</p>}
                {p.confidence_notes && <p className="notes">{p.confidence_notes}</p>}
              </li>
            ))}
          </ul>
          <details
            className="raw-debug"
            open={rawPanelOpen}
            onToggle={(e) => setRawPanelOpen(e.currentTarget.open)}
          >
            <summary>Raw model response (JSON / text)</summary>
            <p className="raw-debug-note">
              Exact assistant message from Ollama. Shown only after a successful API call; request failures appear in
              the red error box above.
            </p>
            <div className="raw-debug-toolbar">
              <button
                type="button"
                className="btn secondary btn-copy"
                onClick={() => copyPattern('__raw__', result.raw_model_text)}
                disabled={!result.raw_model_text.trim()}
              >
                {copiedEntity === '__raw__' ? 'Copied' : 'Copy raw'}
              </button>
            </div>
            <pre className="raw">{prettyModelRaw(result.raw_model_text)}</pre>
          </details>
        </section>
      )}

      <section className="card">
        <h2>5. Validation</h2>
        <p className="hint">
          Upload a similar document and check whether the generated patterns extract the intended entities.
        </p>
        <label className="file">
          <input type="file" accept="application/pdf" onChange={onValidationFile} disabled={busy} />
          <span>{validationUpload ? `Validation: ${validationUpload.filename}` : 'Choose validation PDF'}</span>
        </label>
        {validationUpload && (
          <p className="meta">
            {validationUpload.pages} page(s) · {validationUpload.extraction_method}
            {validationUpload.full_text.length > 0 && ` · ${validationUpload.full_text.length} characters`}
          </p>
        )}
        <div className="actions">
          <button type="button" className="btn primary" onClick={runValidate} disabled={busy || !validationUpload}>
            Validate patterns
          </button>
          <button
            type="button"
            className="btn secondary"
            onClick={() => loadValidationOcrPage(validationPage)}
            disabled={busy || !validationUpload}
          >
            Load boxes
          </button>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showValidationHighlights}
              onChange={(e) => setShowValidationHighlights(e.target.checked)}
            />{' '}
            Highlight matches
          </label>
        </div>

        {validationUpload && (
          <div className="row">
            <label>
              Boxes DPI
              <input
                type="number"
                min={100}
                max={400}
                step={50}
                value={validationBoxesDpi}
                onChange={(e) => setValidationBoxesDpi(Number(e.target.value) || 200)}
                disabled={busy}
              />
            </label>
          </div>
        )}

        {validationUpload && validationOcrView && (
          <div className="annotate">
            <div className="annotate-toolbar">
              <button
                type="button"
                className="btn secondary"
                onClick={() => loadValidationOcrPage(Math.max(1, validationPage - 1))}
                disabled={busy || validationPage <= 1}
              >
                Prev page
              </button>
              <div className="annotate-page">
                Page {validationPage} / {validationUpload.pages}
              </div>
              <button
                type="button"
                className="btn secondary"
                onClick={() => loadValidationOcrPage(Math.min(validationUpload.pages, validationPage + 1))}
                disabled={busy || validationPage >= validationUpload.pages}
              >
                Next page
              </button>
            </div>

            <div className="ocr-canvas" style={{ aspectRatio: `${validationOcrView.width} / ${validationOcrView.height}` }}>
              <img
                className="ocr-image"
                src={`data:image/png;base64,${validationOcrView.image_base64}`}
                alt={`Validation OCR page ${validationOcrView.page}`}
              />
              {validationOcrView.boxes.map((b) => {
                const left = (b.x0 / validationOcrView.width) * 100
                const top = (b.y0 / validationOcrView.height) * 100
                const w = ((b.x1 - b.x0) / validationOcrView.width) * 100
                const h = ((b.y1 - b.y0) / validationOcrView.height) * 100
                const isHit = showValidationHighlights && highlightBoxIds.has(b.id)
                return (
                  <div
                    key={`v-${b.id}`}
                    className={`ocr-box ${isHit ? 'highlight' : ''}`}
                    style={{ left: `${left}%`, top: `${top}%`, width: `${w}%`, height: `${h}%` }}
                    title={b.text}
                  />
                )
              })}
            </div>
          </div>
        )}

        {validation && (
          <div className="validation">
            {Object.entries(validation.matches).map(([entity, vals]) => (
              <div key={entity} className="validation-row">
                <div className="validation-entity">{entity}</div>
                {validation.errors[entity] ? (
                  <div className="validation-error">Regex error: {validation.errors[entity]}</div>
                ) : (
                  <div className="validation-values">
                    {vals && vals.length ? vals.join(' · ') : '(no matches)'}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </section>

        <footer className="footer">
          <p>
            Local Ollama required. Models: <code>ollama list</code>. API: <code>uvicorn</code> in <code>backend</code>.
          </p>
        </footer>
      </div>
    </div>
  )
}
