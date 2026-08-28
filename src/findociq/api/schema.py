"""Typed public HTTP contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from findociq.reason.schema import ExtractedFigure, SourceCitation
from findociq.service import ReasoningMode


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    mode: ReasoningMode | None = None
    question_id: str | None = Field(default=None, min_length=1, max_length=200)
    document_ids: tuple[str, ...] = Field(default=(), max_length=20)


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: ReasoningMode
    answer: str
    citations: tuple[SourceCitation, ...] = Field(min_length=1)


class ExtractRequest(BaseModel):
    """Public request contract for structured, citation-grounded extraction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    question_id: str | None = Field(default=None, min_length=1, max_length=200)
    document_ids: tuple[str, ...] = Field(default=(), max_length=20)


class IngestDocumentRequest(BaseModel):
    """Versioned PDF upload contract for trusted local pipeline callers."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    contract_version: Literal["1.0"] = "1.0"
    filename: str = Field(min_length=5, max_length=240, pattern=r"^[^/\\]+\.[Pp][Dd][Ff]$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_base64: str = Field(min_length=8)


class IngestDocumentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_version: Literal["1.0"] = "1.0"
    document_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    chunk_ids: tuple[str, ...] = Field(min_length=1)
    config_hash: str


class IngestBatchRequest(BaseModel):
    """One GPU lease covering every document in a borrower upload."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0"] = "1.0"
    batch_id: str | None = Field(
        default=None, min_length=3, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$"
    )
    documents: tuple[IngestDocumentRequest, ...] = Field(min_length=1)


class IngestBatchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_version: Literal["1.0"] = "1.0"
    documents: tuple[IngestDocumentResponse, ...] = Field(min_length=1)


class IngestionActivityEvent(BaseModel):
    """Sanitized progress emitted while a document batch owns the GPU."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=1)
    batch_id: str
    stage: str
    message: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_name: str | None = None
    document_index: int | None = Field(default=None, ge=1)
    document_count: int | None = Field(default=None, ge=1)
    completed: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=1)


class IngestionActivityResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    status: Literal["pending", "running", "completed", "failed"]
    events: tuple[IngestionActivityEvent, ...] = ()
    last_sequence: int = Field(ge=0)


class ExtractResponse(BaseModel):
    """Versioned black-box contract consumed by downstream applications."""

    model_config = ConfigDict(frozen=True)

    contract_version: Literal["1.0"] = "1.0"
    question: str
    figures: tuple[ExtractedFigure, ...] = Field(min_length=1)
    notes: tuple[str, ...] = ()


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    service: str = "findociq"
