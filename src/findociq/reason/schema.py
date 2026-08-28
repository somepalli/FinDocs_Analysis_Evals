"""Typed contracts for extracted figures and cited answers."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from findociq.ingest.schema import BoundingBox, Provenance


class SourceCitation(BaseModel):
    """A page-level citation that can be rendered back to the filing."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    page_number: int = Field(ge=1)
    bbox: BoundingBox


class ExtractedFigure(BaseModel):
    """One figure or explicitly extracted fact with grounded provenance."""

    model_config = ConfigDict(frozen=True)

    label: str
    value: str
    unit: str | None = None
    period: str | None = None
    citation: SourceCitation


class Pass1Extraction(BaseModel):
    """Structured output from pass 1; no free-form answer is generated here."""

    model_config = ConfigDict(frozen=True)

    question: str
    figures: tuple[ExtractedFigure, ...] = ()
    notes: tuple[str, ...] = ()


class ReasonedAnswer(BaseModel):
    """An analyst-facing answer whose citations are mandatory."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    citations: tuple[SourceCitation, ...] = Field(min_length=1)


class Pass2EvidenceSelection(BaseModel):
    """Pass-2 output that refers to canonical pass-1 evidence by stable ID."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class ReasoningRun(BaseModel):
    """Result envelope used to compare single- and two-pass runs."""

    model_config = ConfigDict(frozen=True)

    mode: str
    question: str
    answer: ReasonedAnswer
    extraction: Pass1Extraction | None = None


def citation_identity(citation: SourceCitation) -> tuple[object, ...]:
    return (
        citation.document_id,
        citation.page_number,
        citation.bbox.x0,
        citation.bbox.y0,
        citation.bbox.x1,
        citation.bbox.y1,
    )


def provenance_identity(provenance: Provenance) -> tuple[object, ...]:
    return (
        provenance.document_id,
        provenance.page_number,
        provenance.bbox.x0,
        provenance.bbox.y0,
        provenance.bbox.x1,
        provenance.bbox.y1,
    )


def ground_citation(
    citation: SourceCitation,
    candidates: Iterable[SourceCitation],
    *,
    minimum_iou: float = 0.95,
) -> SourceCitation:
    matching_page = [
        candidate
        for candidate in candidates
        if candidate.document_id == citation.document_id
        and candidate.page_number == citation.page_number
    ]
    if not matching_page:
        raise ValueError("citation document/page is not present in the allowed evidence")
    best = max(matching_page, key=lambda candidate: _bbox_iou(citation.bbox, candidate.bbox))
    if _bbox_iou(citation.bbox, best.bbox) < minimum_iou:
        raise ValueError("citation bbox does not match the allowed evidence")
    return best


def citation_from_provenance(provenance: Provenance) -> SourceCitation:
    return SourceCitation(
        document_id=provenance.document_id,
        page_number=provenance.page_number,
        bbox=provenance.bbox,
    )


def _bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    x0 = max(first.x0, second.x0)
    y0 = max(first.y0, second.y0)
    x1 = min(first.x1, second.x1)
    y1 = min(first.y1, second.y1)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = (first.x1 - first.x0) * (first.y1 - first.y0)
    second_area = (second.x1 - second.x0) * (second.y1 - second.y0)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
