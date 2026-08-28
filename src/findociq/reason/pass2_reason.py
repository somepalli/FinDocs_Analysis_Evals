"""Pass 2: reason only over pass-1 structured extraction."""

from __future__ import annotations

from findociq.observability.recorder import TraceObserver
from findociq.observability.schema import TraceContext
from findociq.reason.generation import GenerationClient
from findociq.reason.pass1_extract import _parse_json
from findociq.reason.prompting import load_prompt, render_extraction, substitute
from findociq.reason.schema import (
    Pass1Extraction,
    Pass2EvidenceSelection,
    ReasonedAnswer,
    SourceCitation,
)


class Pass2Reasoner:
    def __init__(self, client: GenerationClient, observer: TraceObserver | None = None) -> None:
        self.client = client
        self.observer = observer or TraceObserver()

    def reason(
        self,
        question: str,
        extraction: Pass1Extraction,
        *,
        trace_context: TraceContext | None = None,
    ) -> ReasonedAnswer:
        if not question.strip():
            raise ValueError("question must not be blank")
        rendered_extraction = render_extraction(extraction)
        prompt = substitute(
            load_prompt("pass2_reason.txt"),
            QUESTION=question,
            EXTRACTION=rendered_extraction,
        )
        raw = self.client.complete(
            prompt, trace_context=trace_context, stage="generation.pass2"
        )
        allowed = {
            f"evidence_{index}": figure.citation
            for index, figure in enumerate(extraction.figures, start=1)
        }
        if not allowed:
            raise ValueError("pass 2 cannot produce a cited answer without extracted figures")
        try:
            selection = self._validate_selection(raw, allowed)
        except ValueError as first_error:
            retry_prompt = substitute(
                load_prompt("pass2_retry.txt"),
                QUESTION=question,
                EXTRACTION=rendered_extraction,
                INVALID_RESPONSE=raw,
                ERROR=str(first_error),
            )
            retry_raw = self.client.complete(
                retry_prompt,
                trace_context=trace_context,
                stage="generation.pass2.retry",
            )
            try:
                selection = self._validate_selection(retry_raw, allowed)
            except ValueError as retry_error:
                raise ValueError(
                    "pass 2 returned an invalid evidence selection after retry"
                ) from retry_error
        context = trace_context or TraceContext.for_query(question, operation="reasoning:pass2")
        with self.observer.span(
            context,
            "citation_validation",
            {"mode": "pass2", "citation_count": len(selection.evidence_ids)},
        ):
            citations = tuple(allowed[evidence_id] for evidence_id in selection.evidence_ids)
        return ReasonedAnswer(answer=selection.answer, citations=citations)

    @staticmethod
    def _validate_selection(
        raw: str, allowed: dict[str, SourceCitation]
    ) -> Pass2EvidenceSelection:
        selection = Pass2EvidenceSelection.model_validate(_parse_json(raw))
        unknown = set(selection.evidence_ids).difference(allowed)
        if unknown:
            invalid = ", ".join(sorted(unknown))
            raise ValueError(f"pass 2 selected unknown evidence IDs: {invalid}")
        unique_ids = tuple(dict.fromkeys(selection.evidence_ids))
        return selection.model_copy(update={"evidence_ids": unique_ids})
