"""Deterministic fail-closed validation before human-review handoff."""

from __future__ import annotations

from fundermatch.matching.schema import RuleGatedRetrievalResult
from fundermatch.orchestration.graph import WorkerContext, WorkerFailure
from fundermatch.orchestration.schema import (
    ApplicationMemoryState,
    GuardrailResult,
    WorkerName,
    WorkerResult,
    result_hash,
)
from fundermatch.orchestration.workspace import ApplicationWorkspace
from fundermatch.rules.schema import BorrowerApplication
from fundermatch.suggest.schema import SuggestionBundle


class GuardrailWorker:
    name = WorkerName.GUARDRAILS

    def __init__(self, workspace: ApplicationWorkspace) -> None:
        self.workspace = workspace

    async def run(self, state: ApplicationMemoryState, context: WorkerContext) -> WorkerResult:
        context.consume_tool_call()
        application = self.workspace.load(
            state.application_id, WorkerName.FINANCIAL_ANALYSIS.value, BorrowerApplication
        )
        retrieval = self.workspace.load(
            state.application_id,
            WorkerName.PRECEDENT_RETRIEVAL.value,
            RuleGatedRetrievalResult,
        )
        suggestion = self.workspace.load(
            state.application_id, WorkerName.SUGGESTION.value, SuggestionBundle
        )
        if application.application_id != state.application_id:
            self._terminal("cross_application_artifact", "Application artifact ownership mismatch")
        if retrieval.application_id != state.application_id:
            self._terminal("cross_application_retrieval", "Retrieval artifact ownership mismatch")
        if suggestion.application.application_id != state.application_id:
            self._terminal("cross_application_suggestion", "Suggestion artifact ownership mismatch")

        document_ids = {item.document_id for item in state.documents}
        if not state.evidence:
            self._attention("missing_evidence", "Cited financial evidence is required")
        if any(item.document_id not in document_ids for item in state.evidence):
            self._terminal("cross_application_evidence", "Evidence does not belong to this application")
        expected_values = {
            "annual_revenue_crore": str(application.profile.annual_revenue_crore),
            "ebitda_margin_pct": str(application.profile.ebitda_margin_pct),
            "dscr": str(application.profile.dscr),
        }
        for evidence in state.evidence:
            if evidence.metric in expected_values and evidence.value != expected_values[evidence.metric]:
                self._attention("unsupported_numeric_claim", "A numeric claim does not resolve to cited evidence")

        eligible = {item.funder_id for item in state.eligibility if item.eligible}
        retrieved = {item.precedent.decision.funder_id for item in retrieval.matches}
        suggested = {item.funder_id for item in suggestion.candidates}
        if not retrieved <= eligible or not suggested <= eligible:
            self._terminal("ineligible_funder_leakage", "An ineligible funder entered retrieval or suggestions")
        if suggestion.authority != "advisory_only" or not suggestion.requires_human_decision:
            self._terminal("automatic_lending_decision", "Suggestion attempted to assume human authority")
        if not suggestion.advisory_notice:
            self._attention("missing_advisory_notice", "Suggestion is missing the human-authority notice")

        results = (
            GuardrailResult(guardrail="citation_presence", passed=True),
            GuardrailResult(guardrail="evidence_ownership", passed=True),
            GuardrailResult(guardrail="numeric_grounding", passed=True),
            GuardrailResult(guardrail="eligible_only", passed=True),
            GuardrailResult(guardrail="human_authority", passed=True),
            GuardrailResult(guardrail="checkpoint_schema", passed=True),
        )
        return WorkerResult(
            worker=self.name,
            output_sha256=result_hash(self.name, state.application_id, *(item.guardrail for item in results)),
            guardrails=results,
        )

    @staticmethod
    def _attention(code: str, message: str) -> None:
        raise WorkerFailure(code, message, needs_attention=True)

    @staticmethod
    def _terminal(code: str, message: str) -> None:
        raise WorkerFailure(code, message, retryable=False)
