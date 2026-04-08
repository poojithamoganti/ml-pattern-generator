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

type EntityForm = {
  id: string
  name: string
  kind: string
  occurrence: 'single' | 'multiple'
  hints: string
  picked_landmark?: { text: string; box_id: string; page: number }
  picked_label?: { text: string; box_id: string; page: number }
  picked_value?: { text: string; box_id: string; page: number }
  examples?: Array<{
    landmark?: { text: string; box_id: string; page: number }
    label?: { text: string; box_id: string; page: number }
    value?: { text: string; box_id: string; page: number }
  }>
}

function toSpecs(rows: EntityForm[]): EntitySpec[] {
  return rows
    .filter((r) => r.name.trim())
    .map((r) => ({
      name: r.name.trim(),
      kind: r.kind || 'text',
      occurrence: r.occurrence || 'single',
      hints: r.hints.trim(),
      examples: (r.examples || []).map((ex) => ({
        landmark: ex.landmark?.text,
        label: ex.label?.text,
        value: ex.value?.text,
      })),
    }))
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

export default function App() {
  const [upload, setUpload] = useState<UploadResponse | null>(null)
  /** Extra PDFs (text only) sent with generate for cross-document pattern synthesis */
  const [extraSamples, setExtraSamples] = useState<Array<{ id: string; filename: string; full_text: string }>>([])
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
          { id: crypto.randomUUID(), filename: u.filename, full_text: u.full_text },
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

  const removeExtraSample = useCallback((id: string) => {
    setExtraSamples((prev) => prev.filter((s) => s.id !== id))
  }, [])

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

  const loadOcrPage = async (page: number) => {
    if (!upload) return
    setErr(null)
    setBusy(true)
    setBusyHint('Loading OCR boxes…')
    try {
      const v = await getOcrBoxes({ upload_id: upload.upload_id, page, dpi: ocrDpiAnnotate })
      setOcrPage(page)
      setOcrView(v)
    } catch (e: unknown) {
      setErr(String(e))
    } finally {
      setBusy(false)
      setBusyHint('')
    }
  }

  const onPickBox = (box: OcrBox) => {
    if (!activePick) return
    setEntities((prev) =>
      prev.map((r) => {
        if (r.id !== activePick.entityId) return r
        if (activePick.field === 'landmark') {
          const nextHints = r.hints.includes('Landmark:')
            ? r.hints
            : `${r.hints}${r.hints.trim() ? '\n' : ''}Landmark: "${box.text}"`
          return { ...r, picked_landmark: { text: box.text, box_id: box.id, page: box.page }, hints: nextHints }
        }
        if (activePick.field === 'label') {
          const nextHints = r.hints.includes('"')
            ? r.hints
            : `${r.hints}${r.hints.trim() ? '\n' : ''}Label text: "${box.text}"`
          return { ...r, picked_label: { text: box.text, box_id: box.id, page: box.page }, hints: nextHints }
        }
        const nextHints =
          r.hints.trim().length === 0
            ? `Example value: "${box.text}"`
            : `${r.hints}\nExample value: "${box.text}"`
        return { ...r, picked_value: { text: box.text, box_id: box.id, page: box.page }, hints: nextHints }
      }),
    )
    setActivePick(null)
  }

  const addEntity = () => {
    setEntities((prev) => [
      ...prev,
      { id: crypto.randomUUID(), name: '', kind: 'text', occurrence: 'single', hints: '' },
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
                onChange={(e) => setOcrDpiAnnotate(Number(e.target.value) || 200)}
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
                  onClick={() => setExtraSamples([])}
                  disabled={busy}
                >
                  Clear extra samples
                </button>
              )}
            </div>
            <p className="muted small">
              Extra PDFs are only used as additional OCR text for &quot;Generate patterns&quot; — same entities, different
              layouts (e.g. alternate New Balance lines).
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

        {docMode === 'annotate' && upload && (
          <div className="annotate">
            <div className="annotate-toolbar">
              <button
                type="button"
                className="btn secondary"
                onClick={() => loadOcrPage(Math.max(1, ocrPage - 1))}
                disabled={busy || ocrPage <= 1}
              >
                Prev page
              </button>
              <div className="annotate-page">
                Page {ocrPage} / {upload.pages}
              </div>
              <button
                type="button"
                className="btn secondary"
                onClick={() => loadOcrPage(Math.min(upload.pages, ocrPage + 1))}
                disabled={busy || ocrPage >= upload.pages}
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
