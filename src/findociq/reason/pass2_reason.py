"""Pass 2: reason only over pass-1 structured extraction."""

from __future__ import annotations

import json

from findociq.observability.recorder import TraceObserver
from findociq.observability.schema import TraceContext
from findociq.reason.generation import GenerationClient
from findociq.reason.pass1_extract import (
    _numeric_groups,
    _numeric_variants,
    _parse_json,
    _unsupported_answer_terms,
)
from findociq.reason.prompting import load_prompt, render_extraction
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
        context = trace_context or TraceContext.for_query(question, operation="reasoning:pass2")
        rendered_extraction = render_extraction(extraction)
        template = load_prompt("pass2_reason.txt")
        prompt = json.dumps(
            {"question": question, "structured_extraction": rendered_extraction},
            ensure_ascii=False,
        )
        raw = self.client.complete(
            prompt,
            system_prompt=template,
            trace_context=context.with_prompt("pass2_reason", template),
            stage="generation.pass2",
        )
        allowed = {
            f"evidence_{index}": figure.citation
            for index, figure in enumerate(extraction.figures, start=1)
        }
        if not allowed:
            raise ValueError("pass 2 cannot produce a cited answer without extracted figures")
        try:
            selection = self._validate_selection(raw, allowed)
            self._validate_answer_support(selection, extraction, question)
        except ValueError as first_error:
            retry_template = load_prompt("pass2_retry.txt")
            retry_prompt = json.dumps(
                {
                    "question": question,
                    "structured_extraction": rendered_extraction,
                    "invalid_response": raw,
                    "validation_error": str(first_error),
                },
                ensure_ascii=False,
            )
            retry_raw = self.client.complete(
                retry_prompt,
                system_prompt=retry_template,
                trace_context=context.with_prompt("pass2_retry", retry_template),
                stage="generation.pass2.retry",
            )
            try:
                selection = self._validate_selection(retry_raw, allowed)
                self._validate_answer_support(selection, extraction, question)
            except ValueError as retry_error:
                raise ValueError(
                    "pass 2 returned an invalid evidence selection after retry"
                ) from retry_error
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

    @staticmethod
    def _validate_answer_support(
        selection: Pass2EvidenceSelection,
        extraction: Pass1Extraction,
        question: str,
    ) -> None:
        figures = {
            f"evidence_{index}": figure
            for index, figure in enumerate(extraction.figures, start=1)
        }
        selected = tuple(figures[item] for item in selection.evidence_ids)
        evidence_text = " ".join(
            " ".join(
                value
                for value in (figure.label, figure.value, figure.unit, figure.period)
                if value
            )
            for figure in selected
        )
        answer_groups = _numeric_groups(selection.answer)
        evidence_numbers = _numeric_variants(evidence_text)
        if answer_groups and not all(
            group.intersection(evidence_numbers) for group in answer_groups
        ):
            raise ValueError("pass 2 answer contains a number unsupported by selected evidence")
        if not answer_groups and not any(
            figure.value.casefold() in selection.answer.casefold() for figure in selected
        ):
            raise ValueError("pass 2 answer is not supported by selected evidence")
        unsupported = _unsupported_answer_terms(selection.answer, question, evidence_text)
        if unsupported:
            raise ValueError(
                "pass 2 answer contains factual terms unsupported by selected evidence"
            )
