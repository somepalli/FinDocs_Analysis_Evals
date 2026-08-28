from __future__ import annotations

import asyncio

import pytest

from fundermatch.matching.schema import RuleGatedRetrievalResult
from fundermatch.orchestration.graph import WorkerContext, WorkerFailure
from fundermatch.orchestration.guardrails import GuardrailWorker
from fundermatch.orchestration.schema import (
    ApplicationMemoryState,
    BoundingBox,
    DocumentReference,
    EligibilityReference,
    EvidenceReference,
    WorkerName,
)
from fundermatch.orchestration.workspace import ApplicationWorkspace
from fundermatch.rules.engine import EligibilityEngine
from fundermatch.rules.schema import BorrowerApplication, FunderPolicy
from fundermatch.suggest.assembler import SuggestionAssembler


def _guardrail_state(
    tmp_path,  # type: ignore[no-untyped-def]
    application: BorrowerApplication,
    policies: tuple[FunderPolicy, ...],
) -> tuple[GuardrailWorker, ApplicationMemoryState]:
    workspace = ApplicationWorkspace(tmp_path)
    eligibility = EligibilityEngine().evaluate_all(application, policies)
    retrieval = RuleGatedRetrievalResult(
        application_id=application.application_id,
        eligibility=eligibility,
        matches=(),
    )
    suggestion = SuggestionAssembler().assemble(application, policies, retrieval)
    workspace.save(application.application_id, WorkerName.FINANCIAL_ANALYSIS.value, application)
    workspace.save(application.application_id, WorkerName.PRECEDENT_RETRIEVAL.value, retrieval)
    workspace.save(application.application_id, WorkerName.SUGGESTION.value, suggestion)
    evidence = tuple(
        EvidenceReference(
            evidence_id=f"evidence-{index}",
            metric=item.name,
            value=str(item.value),
            unit=item.unit,
            period=item.period,
            document_id=item.citation.document_id,
            page_number=item.citation.page_number,
            bbox=BoundingBox.model_validate(item.citation.bbox.model_dump()),
        )
        for index, item in enumerate(application.evidence, start=1)
    )
    documents = tuple(
        DocumentReference(
            document_id=document_id,
            sha256="a" * 64,
            page_count=10,
            chunk_count=20,
        )
        for document_id in sorted({item.document_id for item in evidence})
    )
    references = tuple(
        EligibilityReference(
            funder_id=item.funder_id,
            eligible=item.eligible,
            failed_criteria=tuple(
                check.criterion.value for check in item.checks if not check.passed
            ),
        )
        for item in eligibility
    )
    initial = ApplicationMemoryState.initial(
        application.application_id, worker_order=(WorkerName.GUARDRAILS,)
    )
    state = initial.model_copy(
        update={"documents": documents, "evidence": evidence, "eligibility": references}
    )
    return GuardrailWorker(workspace), state


def test_guardrail_terminally_rejects_cross_application_evidence(
    tmp_path,
    aligned_application: BorrowerApplication,
    funder_policies: tuple[FunderPolicy, ...],
) -> None:  # type: ignore[no-untyped-def]
    worker, state = _guardrail_state(tmp_path, aligned_application, funder_policies)
    poisoned = state.model_copy(
        update={
            "evidence": (
                state.evidence[0].model_copy(update={"document_id": "other-application-doc"}),
                *state.evidence[1:],
            )
        }
    )
    context = WorkerContext(
        application_id=state.application_id,
        command_id=state.command_id_for(WorkerName.GUARDRAILS),
    )
    with pytest.raises(WorkerFailure) as captured:
        asyncio.run(worker.run(poisoned, context))
    assert captured.value.code == "cross_application_evidence"
    assert captured.value.retryable is False


def test_guardrail_terminally_rejects_candidate_not_in_eligible_set(
    tmp_path,
    aligned_application: BorrowerApplication,
    funder_policies: tuple[FunderPolicy, ...],
) -> None:  # type: ignore[no-untyped-def]
    worker, state = _guardrail_state(tmp_path, aligned_application, funder_policies)
    poisoned = state.model_copy(
        update={
            "eligibility": tuple(
                item.model_copy(update={"eligible": False}) for item in state.eligibility
            )
        }
    )
    context = WorkerContext(
        application_id=state.application_id,
        command_id=state.command_id_for(WorkerName.GUARDRAILS),
    )
    with pytest.raises(WorkerFailure) as captured:
        asyncio.run(worker.run(poisoned, context))
    assert captured.value.code == "ineligible_funder_leakage"
    assert captured.value.retryable is False
