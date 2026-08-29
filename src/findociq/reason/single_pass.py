"""Single-pass baseline used for Phase 4 comparison."""

from __future__ import annotations

from findociq.observability.recorder import TraceObserver
from findociq.observability.schema import TraceContext
from findociq.reason.generation import GenerationClient
from findociq.reason.pass1_extract import _parse_json
from findociq.reason.prompting import load_prompt, render_evidence, substitute
from findociq.reason.schema import ReasonedAnswer, citation_from_provenance, ground_citation
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
                    substitute(
                        template,
                        QUESTION=question,
                        EVIDENCE=render_evidence(hits),
                    ),
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
        return answer.model_copy(update={"citations": citations})
