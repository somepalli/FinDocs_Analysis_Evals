"""Typed public HTTP contracts."""

from __future__ import annotations

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
