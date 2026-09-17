"""Content-based evidence routing. Classification does not certify authenticity."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from findociq.ingest.schema import Chunk
from findociq.reason.schema import SourceCitation, citation_from_provenance

DocumentType = Literal[
    "pan",
    "aadhaar",
    "gst",
    "udyam",
    "incorporation",
    "agreement",
    "bank_statement",
    "utility_bill",
    "audited_financials",
    "financial_statement",
    "financial_information_report",
    "payroll",
    "valuation",
    "tax_return",
    "no_dues",
    "unknown",
    "mixed",
]


class ClassificationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str
    patterns: dict[DocumentType, tuple[str, ...]]

    @classmethod
    def load(cls, path: Path = Path("configs/evidence/classification.yaml")):
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class SectionClassification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    document_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    types: tuple[DocumentType, ...]
    status: Literal["confirmed", "ambiguous", "unknown"]
    citations: tuple[SourceCitation, ...] = ()
    classifier_version: str


def classify_sections(chunks: tuple[Chunk, ...], policy: ClassificationPolicy):
    pages = defaultdict(list)
    for chunk in chunks:
        owners = {p.document_id for p in chunk.provenance}
        if len(owners) != 1:
            raise ValueError("evidence_scope_violation")
        for p in chunk.provenance:
            pages[(p.document_id, p.page_number)].append(chunk)
    sections = []
    for (document, page), items in sorted(pages.items()):
        hits = {
            name: tuple(c for c in items if any(re.search(p, c.text, re.I) for p in patterns))
            for name, patterns in policy.patterns.items()
        }
        hits = {name: items for name, items in hits.items() if items}
        if "financial_information_report" in hits:
            hits.pop("audited_financials", None)
        # A more specific audited heading also contains the generic statement heading.
        if "audited_financials" in hits:
            hits.pop("financial_statement", None)
        status = "confirmed" if len(hits) == 1 else "ambiguous" if hits else "unknown"
        citations = {
            p.model_dump_json(): citation_from_provenance(p)
            for items in hits.values()
            for c in items
            for p in c.provenance
            if p.document_id == document and p.page_number == page
        }
        section = SectionClassification(
            document_id=document,
            page_start=page,
            page_end=page,
            types=tuple(sorted(hits)) or ("unknown",),
            status=status,
            citations=tuple(citations.values()),
            classifier_version=policy.version,
        )
        if (
            sections
            and sections[-1].document_id == document
            and sections[-1].page_end == page - 1
            and sections[-1].types == section.types
            and sections[-1].status == status
        ):
            old = sections.pop()
            section = old.model_copy(
                update={"page_end": page, "citations": old.citations + section.citations}
            )
        sections.append(section)
    return tuple(sections)
