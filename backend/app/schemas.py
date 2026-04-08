from pydantic import BaseModel, Field


class EntitySpec(BaseModel):
    name: str = Field(..., min_length=1, description="Entity label, e.g. Invoice Number")
    kind: str = Field(
        default="text",
        description="Expected value shape: text, date, amount, currency, number, email, phone, address, id, other",
    )
    hints: str = Field(
        default="",
        description="Where it appears on the page, layout, column, or format — not required to paste exact text",
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
    filename: str
    pages: int
    text_preview: str
    full_text: str
    extraction_method: str
    extraction_mode: str = ""
    ocr_engine: str = ""
    ocr_dpi: int = 300
