"""Single-pass baseline used for Phase 4 comparison."""

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
from findociq.reason.prompting import load_prompt, render_evidence
from findociq.reason.schema import (
    ReasonedAnswer,
    citation_from_provenance,
    citation_identity,
    ground_citation,
)
from findociq.retrieve.schema import RetrievalHit


class SinglePassReasoner:
    def __init__(self, client: GenerationClient, observer: TraceObserver | None = None) -> None:
        self.client = client
        self.observer = observer or TraceObserver()

    def reason(
        self,
        question: str,
        hits: tuple[RetrievalHit, ...],
        *,
        trace_context: TraceContext | None = None,
    ) -> ReasonedAnswer:
        if not question.strip():
            raise ValueError("question must not be blank")
        if not hits:
            raise ValueError("single-pass reasoning requires retrieved evidence")
        context = trace_context or TraceContext.for_query(
            question, operation="reasoning:single_pass"
        )
        template = load_prompt("single_pass_reason.txt")
        prompt_context = context.with_prompt("single_pass_reason", template)
        answer = ReasonedAnswer.model_validate(
            _parse_json(
                self.client.complete(
                    json.dumps(
                        {"question": question, "untrusted_evidence": render_evidence(hits)},
                        ensure_ascii=False,
                    ),
                    system_prompt=template,
                    trace_context=prompt_context,
                    stage="generation.single_pass",
                )
            )
        )
        allowed = tuple(
            citation_from_provenance(provenance)
            for hit in hits
            for provenance in hit.chunk.provenance
        )
        with self.observer.span(
            context,
            "citation_validation",
            {"mode": "single_pass", "citation_count": len(answer.citations)},
        ):
            try:
                citations = tuple(
                    ground_citation(citation, allowed) for citation in answer.citations
                )
            except ValueError as error:
                raise ValueError(
                    "single-pass returned a citation not present in retrieved evidence"
                ) from error
        cited_text = " ".join(
            hit.chunk.text
            for hit in hits
            if any(
                citation_identity(citation) == citation_identity(citation_from_provenance(item))
                for citation in citations
                for item in hit.chunk.provenance
            )
        )
        evidence_numbers = _numeric_variants(cited_text)
        if any(
            not group.intersection(evidence_numbers) for group in _numeric_groups(answer.answer)
        ):
            raise ValueError("single-pass answer contains a number unsupported by cited evidence")
        unsupported = _unsupported_answer_terms(answer.answer, question, cited_text)
        if unsupported:
            raise ValueError(
                "single-pass answer contains factual terms unsupported by cited evidence"
            )
        return answer.model_copy(update={"citations": citations})
