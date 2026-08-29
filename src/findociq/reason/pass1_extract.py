"""Pass 1: extract figures and citations into typed JSON."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation

from findociq.observability.recorder import TraceObserver
from findociq.observability.schema import TraceContext
from findociq.reason.generation import GenerationClient
from findociq.reason.prompting import load_prompt, render_evidence, substitute
from findociq.reason.schema import (
    Pass1Extraction,
    SourceCitation,
    citation_from_provenance,
    citation_identity,
    ground_citation,
)
from findociq.retrieve.schema import RetrievalHit


class Pass1Extractor:
    def __init__(self, client: GenerationClient, observer: TraceObserver | None = None) -> None:
        self.client = client
        self.observer = observer or TraceObserver()

    def extract(
        self,
        question: str,
        hits: tuple[RetrievalHit, ...],
        *,
        trace_context: TraceContext | None = None,
    ) -> Pass1Extraction:
        if not question.strip():
            raise ValueError("question must not be blank")
        if not hits:
            raise ValueError("pass 1 requires at least one retrieved evidence chunk")
        context = trace_context or TraceContext.for_query(question, operation="reasoning:pass1")
        template = load_prompt("pass1_extract.txt")
        prompt = substitute(
            template,
            QUESTION=question,
            EVIDENCE=render_evidence(hits),
        )
        raw = self.client.complete(
            prompt,
            trace_context=context.with_prompt("pass1_extract", template),
            stage="generation.pass1",
        )
        payload = _repair_grounded_output(_parse_json(raw), question, hits)
        extraction = Pass1Extraction.model_validate(payload)
        with self.observer.span(
            context,
            "citation_validation",
            {"mode": "pass1", "citation_count": len(extraction.figures)},
        ):
            return self._ground_citations(extraction, hits)

    @staticmethod
    def _ground_citations(
        extraction: Pass1Extraction, hits: tuple[RetrievalHit, ...]
    ) -> Pass1Extraction:
        allowed = tuple(
            citation_from_provenance(provenance)
            for hit in hits
            for provenance in hit.chunk.provenance
        )
        try:
            figures = tuple(
                figure.model_copy(update={"citation": ground_citation(figure.citation, allowed)})
                for figure in extraction.figures
            )
        except ValueError as error:
            raise ValueError(
                "pass 1 returned a citation not present in retrieved evidence"
            ) from error
        return extraction.model_copy(update={"figures": figures})


def _parse_json(raw: str) -> dict[str, object]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("generation response did not contain a JSON object") from None
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("generation response must be a JSON object")
    return value


_NUMBER = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.,])")


def _repair_grounded_output(
    payload: dict[str, object],
    question: str,
    hits: tuple[RetrievalHit, ...],
) -> dict[str, object]:
    """Repair omissions only when the caller or evidence determines the value."""
    repaired = dict(payload)
    repaired["question"] = question
    figures = repaired.get("figures")
    if not isinstance(figures, list):
        return repaired
    repaired_figures: list[object] = []
    for item in figures:
        if not isinstance(item, dict):
            repaired_figures.append(item)
            continue
        citation = _unique_value_citation(
            item.get("value"), hits, proposed=item.get("citation")
        )
        if citation is None:
            citation = _canonical_evidence_citation(item.get("citation"), hits)
        if citation is None:
            repaired_figures.append(item)
            continue
        repaired_item = dict(item)
        repaired_item["citation"] = citation.model_dump(mode="json")
        repaired_figures.append(repaired_item)
    repaired["figures"] = repaired_figures
    return repaired


def _unique_value_citation(
    raw_value: object,
    hits: tuple[RetrievalHit, ...],
    *,
    proposed: object = None,
) -> SourceCitation | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    value_numbers = _normalized_numbers(raw_value)
    matching: list[SourceCitation] = []
    for hit in hits:
        text_matches = (
            bool(value_numbers)
            and len(value_numbers) == 1
            and value_numbers[0] in _normalized_numbers(hit.chunk.text)
        ) or (not value_numbers and raw_value.strip().casefold() in hit.chunk.text.casefold())
        if text_matches:
            matching.extend(citation_from_provenance(item) for item in hit.chunk.provenance)
    identities = {citation_identity(item): item for item in matching}
    if len(identities) == 1:
        return next(iter(identities.values()))
    if not isinstance(proposed, dict):
        return None
    try:
        proposed_citation = SourceCitation.model_validate(proposed)
    except ValueError:
        return None
    matching_page = {
        identity: citation
        for identity, citation in identities.items()
        if citation.document_id == proposed_citation.document_id
        and citation.page_number == proposed_citation.page_number
    }
    if len(matching_page) == 1:
        return next(iter(matching_page.values()))
    if matching_page:
        try:
            return ground_citation(
                proposed_citation, matching_page.values(), minimum_iou=0.01
            )
        except ValueError:
            pass
    return None


def _canonical_evidence_citation(
    proposed: object, hits: tuple[RetrievalHit, ...]
) -> SourceCitation | None:
    """Snap a model bbox only when its retrieved document/page is unambiguous."""
    if not isinstance(proposed, dict):
        return None
    try:
        proposed_citation = SourceCitation.model_validate(proposed)
    except ValueError:
        return None
    candidates = {
        citation_identity(citation): citation
        for hit in hits
        for provenance in hit.chunk.provenance
        if (citation := citation_from_provenance(provenance)).document_id
        == proposed_citation.document_id
        and citation.page_number == proposed_citation.page_number
    }
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if candidates:
        try:
            return ground_citation(
                proposed_citation, candidates.values(), minimum_iou=0.01
            )
        except ValueError:
            pass
    return None


def _normalized_numbers(value: str) -> tuple[Decimal, ...]:
    normalized: list[Decimal] = []
    for token in _NUMBER.findall(value):
        try:
            normalized.append(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return tuple(normalized)
