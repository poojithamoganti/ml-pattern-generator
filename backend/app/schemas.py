from pydantic import BaseModel, Field


class EntitySpec(BaseModel):
    name: str = Field(..., min_length=1, description="Entity label, e.g. Invoice Number")
    kind: str = Field(
        default="text",
        description="Expected value shape: text, date, amount, currency, number, email, phone, address, id, other",
    )
    occurrence: str = Field(
        default="single",
        description='Whether the entity is expected to appear once or multiple times on a page: "single" | "multiple"',
    )
    hints: str = Field(
        default="",
        description="Where it appears on the page, layout, column, or format — not required to paste exact text",
    )
    examples: list[dict[str, str]] = Field(
        default_factory=list,
        description="Optional examples for this entity. Each item may include keys: landmark, label, value.",
    )


class RegexGenerateRequest(BaseModel):
    full_text: str = Field(..., min_length=1, description="OCR/extracted document text")
    entities: list[EntitySpec] = Field(..., min_length=1)
    model: str | None = Field(default=None, description="Ollama model name")
    extra_instructions: str = Field(
        default="",
        description="User hints: dialect, case sensitivity, multiline, etc.",
    )


class RegexPatternItem(BaseModel):
    entity: str
    pattern: str
    flags: str = ""
    rationale: str = ""
    confidence_notes: str = ""


class RegexLlmEnvelope(BaseModel):
    """Shape of the model JSON when using Ollama structured output + Pydantic validation."""

    patterns: list[RegexPatternItem] = Field(
        ...,
        description="One entry per entity from the user list; entity must match the given name exactly",
    )


class RegexGenerateResponse(BaseModel):
    patterns: list[RegexPatternItem]
    raw_model_text: str = ""
    ollama_model: str = ""


class RegexBatchRequest(BaseModel):
    full_text: str = Field(..., min_length=1)
    entities: list[EntitySpec] = Field(..., min_length=1)
    models: list[str] = Field(..., min_length=1, description="Ollama model names to run in parallel")
    extra_instructions: str = ""


class RegexBatchResponse(BaseModel):
    results: list[RegexGenerateResponse]


class UploadResponse(BaseModel):
    upload_id: str = Field(..., description="Server-side id for the uploaded PDF (prefix of stored filename)")
    filename: str
    pages: int
    text_preview: str
    full_text: str
    extraction_method: str
    extraction_mode: str = ""
    ocr_engine: str = ""
    ocr_dpi: int = 300


class OcrBox(BaseModel):
    id: str
    text: str
    page: int
    # Pixel coordinates in the rendered page image (same DPI as requested)
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0


class OcrBoxesResponse(BaseModel):
    upload_id: str
    page: int
    dpi: int
    width: int
    height: int
    # PNG bytes as base64 (data URL is built in frontend)
    image_base64: str
    boxes: list[OcrBox]


class RegexValidateRequest(BaseModel):
    full_text: str = Field(..., min_length=1)
    patterns: list[RegexPatternItem] = Field(..., min_length=1)


class RegexValidateResponse(BaseModel):
    matches: dict[str, list[str]]
    errors: dict[str, str] = Field(default_factory=dict)
