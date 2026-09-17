"""Typed public HTTP contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from findociq.reason.classification import SectionClassification
from findociq.reason.schema import (
    BasisInterpretation,
    ExtractedFigure,
    FieldAssessment,
    SourceCitation,
)
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
    metric_id: str | None = Field(default=None, min_length=1, max_length=100)


class ProductionExtractRequest(BaseModel):
    """Application-scoped production extraction contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    contract_version: Literal["2.0"] = "2.0"
    application_id: str = Field(min_length=3, max_length=200)
    document_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    metric_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    command_id: str = Field(min_length=8, max_length=200)


class DocumentSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    application_id: str = Field(min_length=3, max_length=200)
    document_ids: tuple[str, ...] = Field(min_length=1, max_length=10000)


class DocumentSetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    application_id: str
    document_set_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DocumentSetValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    application_id: str = Field(min_length=3, max_length=200)
    document_set_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class DocumentSetValidationResponse(DocumentSetResponse):
    evidence_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DocumentSetExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["2.1"] = "2.1"
    application_id: str = Field(min_length=3, max_length=200)
    document_set_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    command_id: str = Field(min_length=8, max_length=200)
    interpretations: tuple[BasisInterpretation, ...] = Field(default=(), max_length=20)
    interpretation_command_id: str | None = Field(default=None, pattern=r"^[0-9a-f-]{36}$")
    assessment_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class EvidenceAssessmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    application_id: str
    document_set_id: str
    policy_hash: str
    interpretation_command_id: str | None = None
    fields: tuple[FieldAssessment, ...]
    interpretation_options: tuple[BasisInterpretation, ...] = ()
    classifications: tuple[SectionClassification, ...] = ()


class IngestDocumentRequest(BaseModel):
    """Versioned PDF upload contract for trusted local pipeline callers."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    contract_version: Literal["1.0", "2.0"] = "1.0"
    filename: str = Field(min_length=5, max_length=240, pattern=r"^[^/\\]+\.(?i:pdf|jpe?g|png)$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_base64: str = Field(min_length=8)
    application_id: str | None = Field(default=None, min_length=3, max_length=200)
    command_id: str | None = Field(default=None, min_length=8, max_length=200)
    policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_v2_scope(self) -> IngestDocumentRequest:
        if self.contract_version == "2.0" and not all(
            (self.application_id, self.command_id, self.policy_hash)
        ):
            raise ValueError("v2 ingestion requires application_id, command_id, and policy_hash")
        return self


class IngestDocumentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_version: Literal["1.0", "2.0"] = "1.0"
    document_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    chunk_ids: tuple[str, ...] = Field(min_length=1)
    config_hash: str
    application_id: str | None = None
    policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scan_status: str | None = None
    scan_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dlp_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    storage_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ownership_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class IngestBatchRequest(BaseModel):
    """One GPU lease covering every document in a borrower upload."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0", "2.0"] = "1.0"
    batch_id: str | None = Field(
        default=None, min_length=3, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$"
    )
    documents: tuple[IngestDocumentRequest, ...] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def consistent_contract(self) -> IngestBatchRequest:
        if any(item.contract_version != self.contract_version for item in self.documents):
            raise ValueError("batch and document contract versions must match")
        if self.contract_version == "2.0":
            applications = {item.application_id for item in self.documents}
            if len(applications) != 1:
                raise ValueError("production batch must belong to one application")
        return self


class IngestBatchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_version: Literal["1.0", "2.0"] = "1.0"
    documents: tuple[IngestDocumentResponse, ...] = Field(min_length=1)
    document_set_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    document_set_policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


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

    contract_version: Literal["1.0", "2.0", "2.1"] = "1.0"
    question: str
    figures: tuple[ExtractedFigure, ...] = Field(min_length=1)
    notes: tuple[str, ...] = ()
    application_id: str | None = None
    command_id: str | None = None
    policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    service: str = "findociq"
    policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class RetentionDeleteRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: Literal["2.0"] = "2.0"
    command_id: str = Field(min_length=8, max_length=200)


class RetentionDeleteResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: Literal["2.0"] = "2.0"
    application_id: str
    deleted_document_ids: tuple[str, ...]
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
