"""Fixed, typed workers for the FunderMatch application graph."""

from __future__ import annotations

import asyncio
from base64 import b64encode
from hashlib import sha256
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field

from fundermatch.clients.findociq_client import FinDocIQClient
from fundermatch.clients.findociq_contract import (
    ExtractRequest,
    ExtractedFigure,
    IngestBatchRequest,
    IngestDocumentRequest,
)
from fundermatch.intake import IntakeDocument, _decimal, _default_unit, _select_figure
from fundermatch.matching.retriever import RuleGatedPrecedentRetriever
from fundermatch.matching.schema import RuleGatedRetrievalResult
from fundermatch.orchestration.graph import WorkerContext, WorkerFailure
from fundermatch.orchestration.schema import (
    ApplicationMemoryState,
    BoundingBox,
    DocumentReference,
    EligibilityReference,
    EvidenceReference,
    HumanReviewCheckpoint,
    WorkerName,
    WorkerResult,
    result_hash,
)
from fundermatch.orchestration.workspace import ApplicationWorkspace
from fundermatch.precedent.schema import EvidenceMetric, FinancialProfile
from fundermatch.rules.engine import EligibilityEngine
from fundermatch.rules.schema import BorrowerApplication, FunderEligibility, FunderPolicy
from fundermatch.suggest.assembler import SuggestionAssembler
from fundermatch.suggest.schema import SuggestionBundle
from fundermatch.workflow.errors import WorkflowNotFoundError
from fundermatch.workflow.schema import (
    ActorClaims,
    PipelineAdvanceCommand,
    WorkflowRecord,
    WorkflowState,
)
from fundermatch.workflow.service import WorkflowService


class ActivityReporter(Protocol):
    async def __call__(self, application_id: str, stage: str, **details: object) -> None: ...


class DocumentArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: tuple[IntakeDocument, ...] = Field(min_length=1)
    chunk_ids_by_document: dict[str, tuple[str, ...]]
    config_hashes: dict[str, str]


class MetricArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    figure: ExtractedFigure


class EligibilityArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    application_id: str
    results: tuple[FunderEligibility, ...]


class WorkerDependencies:
    def __init__(
        self,
        *,
        workspace: ApplicationWorkspace,
        findociq: FinDocIQClient,
        workflow: WorkflowService,
        retriever: RuleGatedPrecedentRetriever,
        policies: tuple[FunderPolicy, ...],
        actor: ActorClaims,
        activity: ActivityReporter | None = None,
    ) -> None:
        self.workspace = workspace
        self.findociq = findociq
        self.workflow = workflow
        self.retriever = retriever
        self.policies = policies
        self.actor = actor
        self.activity = activity

    async def report(self, application_id: str, stage: str, **details: object) -> None:
        if self.activity is not None:
            await self.activity(application_id, stage, **details)


class _Worker:
    name: WorkerName

    def __init__(self, dependencies: WorkerDependencies) -> None:
        self.d = dependencies


class DocumentProcessingWorker(_Worker):
    name = WorkerName.DOCUMENT_PROCESSING

    async def run(self, state: ApplicationMemoryState, context: WorkerContext) -> WorkerResult:
        artifact_name = self.name.value
        if self.d.workspace.exists(state.application_id, artifact_name):
            artifact = self.d.workspace.load(state.application_id, artifact_name, DocumentArtifact)
        else:
            request = self.d.workspace.request(state.application_id)
            ingest_documents = []
            for staged in request.files:
                context.consume_tool_call()
                content = self.d.workspace.read_pdf(
                    state.application_id, staged.filename, staged.sha256
                )
                ingest_documents.append(
                    IngestDocumentRequest(
                        filename=staged.filename,
                        sha256=staged.sha256,
                        content_base64=b64encode(content).decode("ascii"),
                    )
                )
            await self.d.report(state.application_id, "document_processing", attempt=state.attempts.get(self.name, 0) + 1)
            context.consume_tool_call()
            try:
                response = await self.d.findociq.ingest_batch(
                    IngestBatchRequest(
                        batch_id=state.job_id or str(context.command_id),
                        documents=tuple(ingest_documents),
                    )
                )
            except Exception as error:
                raise WorkerFailure("findociq_ingest_unavailable", "FinDocIQ ingestion is temporarily unavailable") from error
            expected = {item.filename: item.sha256 for item in request.files}
            if {item.filename: item.sha256 for item in response.documents} != expected:
                raise WorkerFailure("ingest_receipt_mismatch", "FinDocIQ ingestion receipt did not match staged documents", retryable=False)
            artifact = DocumentArtifact(
                documents=tuple(
                    IntakeDocument(
                        filename=item.filename,
                        sha256=item.sha256,
                        document_id=item.document_id,
                        page_count=item.page_count,
                        chunk_count=item.chunk_count,
                    )
                    for item in response.documents
                ),
                chunk_ids_by_document={item.document_id: item.chunk_ids for item in response.documents},
                config_hashes={item.document_id: item.config_hash for item in response.documents},
            )
            self.d.workspace.save(state.application_id, artifact_name, artifact)
        workflow = await _ensure_workflow(self.d, state.application_id, context)
        references = tuple(
            DocumentReference(
                document_id=item.document_id,
                sha256=item.sha256,
                page_count=item.page_count,
                chunk_count=item.chunk_count,
                config_hash=artifact.config_hashes[item.document_id],
            )
            for item in artifact.documents
        )
        return WorkerResult(
            worker=self.name,
            output_sha256=result_hash(self.name, *(item.document_id for item in references)),
            documents=references,
            workflow_state=workflow.state.value,
            workflow_version=workflow.version,
            config_hash=_combined_hash(tuple(artifact.config_hashes.values())),
        )


class FinancialMetricExtractionWorker(_Worker):
    name = WorkerName.FINANCIAL_ANALYSIS
    specs = (
        ("annual_revenue_crore", "Extract the latest annual revenue in INR crore."),
        ("ebitda_margin_pct", "Extract the latest EBITDA margin percentage."),
        ("dscr", "Extract the latest debt service coverage ratio (DSCR)."),
    )

    async def run(self, state: ApplicationMemoryState, context: WorkerContext) -> WorkerResult:
        request = self.d.workspace.request(state.application_id)
        documents = self.d.workspace.load(state.application_id, WorkerName.DOCUMENT_PROCESSING.value, DocumentArtifact)
        document_ids = tuple(item.document_id for item in documents.documents)
        evidence: list[EvidenceMetric] = []
        for metric, question in self.specs:
            artifact_name = f"financial/{metric}"
            if self.d.workspace.exists(state.application_id, artifact_name):
                metric_artifact = self.d.workspace.load(state.application_id, artifact_name, MetricArtifact)
            else:
                context.consume_tool_call()
                await self.d.report(state.application_id, "extracting_metric", metric=metric)
                try:
                    response = await self.d.findociq.extract(
                        ExtractRequest(
                            question=question,
                            question_id=f"{state.application_id}:{metric}",
                            document_ids=document_ids,
                        )
                    )
                except Exception as error:
                    raise WorkerFailure("findociq_extract_unavailable", f"FinDocIQ extraction is temporarily unavailable for {metric}") from error
                metric_artifact = MetricArtifact(metric=metric, figure=_select_figure(response.figures, metric))
                self.d.workspace.save(state.application_id, artifact_name, metric_artifact)
            figure = metric_artifact.figure
            if figure.citation.document_id not in document_ids:
                raise WorkerFailure("cross_application_evidence", "FinDocIQ returned evidence outside the current application", retryable=False)
            evidence.append(
                EvidenceMetric(
                    name=metric,
                    value=_decimal(figure.value),
                    unit=figure.unit or _default_unit(metric),
                    period=figure.period or "latest reported period",
                    citation=figure.citation,
                )
            )
        values = {item.name: item.value for item in evidence}
        metadata = request.metadata
        application = BorrowerApplication(
            application_id=state.application_id,
            borrower_name=metadata.borrower_name,
            industry=metadata.industry,
            region=metadata.region,
            profile=FinancialProfile(
                annual_revenue_crore=values["annual_revenue_crore"],
                requested_amount_crore=metadata.requested_amount_crore,
                ebitda_margin_pct=values["ebitda_margin_pct"],
                dscr=values["dscr"],
                debt_to_ebitda=metadata.debt_to_ebitda,
                collateral_cover=metadata.collateral_cover,
                years_operating=metadata.years_operating,
                employee_count=metadata.employee_count,
            ),
            evidence=tuple(evidence),
            finance_context=metadata.finance_context,
            operations_context=metadata.operations_context,
        )
        digest = self.d.workspace.save(state.application_id, self.name.value, application)
        workflow = await _advance(self.d, state.application_id, WorkflowState.EXTRACTED, context)
        refs = tuple(_evidence_reference(item) for item in evidence)
        return WorkerResult(
            worker=self.name,
            output_sha256=digest,
            evidence=refs,
            workflow_state=workflow.state.value,
            workflow_version=workflow.version,
        )


class DeterministicEligibilityWorker(_Worker):
    name = WorkerName.ELIGIBILITY

    async def run(self, state: ApplicationMemoryState, context: WorkerContext) -> WorkerResult:
        application = self.d.workspace.load(state.application_id, WorkerName.FINANCIAL_ANALYSIS.value, BorrowerApplication)
        results = EligibilityEngine().evaluate_all(application, self.d.policies)
        artifact = EligibilityArtifact(application_id=state.application_id, results=results)
        digest = self.d.workspace.save(state.application_id, self.name.value, artifact)
        workflow = await _advance(self.d, state.application_id, WorkflowState.RULE_GATED, context)
        refs = tuple(
            EligibilityReference(
                funder_id=item.funder_id,
                eligible=item.eligible,
                failed_criteria=tuple(check.criterion.value for check in item.checks if not check.passed),
            )
            for item in results
        )
        return WorkerResult(worker=self.name, output_sha256=digest, eligibility=refs, workflow_state=workflow.state.value, workflow_version=workflow.version)


class EligiblePrecedentRetrievalWorker(_Worker):
    name = WorkerName.PRECEDENT_RETRIEVAL

    async def run(self, state: ApplicationMemoryState, context: WorkerContext) -> WorkerResult:
        context.consume_tool_call()
        application = self.d.workspace.load(state.application_id, WorkerName.FINANCIAL_ANALYSIS.value, BorrowerApplication)
        expected = self.d.workspace.load(state.application_id, WorkerName.ELIGIBILITY.value, EligibilityArtifact)
        try:
            retrieval = await asyncio.to_thread(self.d.retriever.retrieve, application, self.d.policies)
        except RuntimeError as error:
            raise WorkerFailure("ineligible_precedent", "Precedent retrieval returned an ineligible funder", retryable=False) from error
        except Exception as error:
            raise WorkerFailure("qdrant_unavailable", "Precedent retrieval is temporarily unavailable") from error
        if retrieval.eligibility != expected.results:
            raise WorkerFailure("eligibility_changed", "Eligibility result changed before precedent retrieval", needs_attention=True)
        digest = self.d.workspace.save(state.application_id, self.name.value, retrieval)
        return WorkerResult(worker=self.name, output_sha256=digest)


class AdvisorySuggestionWorker(_Worker):
    name = WorkerName.SUGGESTION

    async def run(self, state: ApplicationMemoryState, context: WorkerContext) -> WorkerResult:
        application = self.d.workspace.load(state.application_id, WorkerName.FINANCIAL_ANALYSIS.value, BorrowerApplication)
        retrieval = self.d.workspace.load(state.application_id, WorkerName.PRECEDENT_RETRIEVAL.value, RuleGatedRetrievalResult)
        suggestion = SuggestionAssembler().assemble(application, self.d.policies, retrieval)
        digest = self.d.workspace.save(state.application_id, self.name.value, suggestion)
        workflow = await _advance(
            self.d,
            state.application_id,
            WorkflowState.AI_SUGGESTED,
            context,
            suggestion=suggestion.model_dump(mode="json"),
        )
        return WorkerResult(worker=self.name, output_sha256=digest, workflow_state=workflow.state.value, workflow_version=workflow.version)


class HumanReviewHandoffWorker(_Worker):
    name = WorkerName.HUMAN_REVIEW

    async def run(self, state: ApplicationMemoryState, context: WorkerContext) -> WorkerResult:
        workflow = await _advance(self.d, state.application_id, WorkflowState.AWAITING_HUMAN, context)
        checkpoint = HumanReviewCheckpoint(
            expected_workflow_version=workflow.version,
            allowed_actions=("approve", "reject", "approve_with_conditions", "send_back"),
        )
        return WorkerResult(
            worker=self.name,
            output_sha256=result_hash(self.name, state.application_id, str(workflow.version)),
            human_review=checkpoint,
            workflow_state=workflow.state.value,
            workflow_version=workflow.version,
        )


def fixed_workers(dependencies: WorkerDependencies, guardrail_worker: object) -> tuple[object, ...]:
    """The fixed order is code-owned and cannot be model- or request-selected."""

    return (
        DocumentProcessingWorker(dependencies),
        FinancialMetricExtractionWorker(dependencies),
        DeterministicEligibilityWorker(dependencies),
        EligiblePrecedentRetrievalWorker(dependencies),
        AdvisorySuggestionWorker(dependencies),
        guardrail_worker,
        HumanReviewHandoffWorker(dependencies),
    )


async def _ensure_workflow(dependencies: WorkerDependencies, application_id: str, context: WorkerContext) -> WorkflowRecord:
    try:
        return await dependencies.workflow.get(application_id)
    except WorkflowNotFoundError:
        result = await dependencies.workflow.create(
            application_id,
            dependencies.actor,
            command_id=context.command_id,
            reason="Agent-supervised borrower PDFs accepted into intake",
        )
        return result.workflow


async def _advance(
    dependencies: WorkerDependencies,
    application_id: str,
    target: WorkflowState,
    context: WorkerContext,
    *,
    suggestion: dict[str, object] | None = None,
) -> WorkflowRecord:
    current = await _ensure_workflow(dependencies, application_id, context)
    order = tuple(WorkflowState)
    if order.index(current.state) >= order.index(target):
        return current
    command_id = uuid5(NAMESPACE_URL, f"fundermatch:{application_id}:workflow:{target.value}")
    result = await dependencies.workflow.advance_pipeline(
        application_id,
        PipelineAdvanceCommand(
            command_id=command_id,
            expected_version=current.version,
            target_state=target,
            reason=f"Agent worker completed {target.value.lower()}",
            suggestion=suggestion,
        ),
        dependencies.actor,
    )
    return result.workflow


def _evidence_reference(item: EvidenceMetric) -> EvidenceReference:
    citation = item.citation
    canonical = f"{citation.document_id}|{citation.page_number}|{citation.bbox.model_dump_json()}|{item.name}"
    return EvidenceReference(
        evidence_id=sha256(canonical.encode("utf-8")).hexdigest(),
        metric=item.name,
        value=str(item.value),
        unit=item.unit,
        period=item.period,
        document_id=citation.document_id,
        page_number=citation.page_number,
        bbox=BoundingBox.model_validate(citation.bbox.model_dump()),
    )


def _combined_hash(values: tuple[str, ...]) -> str:
    return sha256("|".join(sorted(values)).encode("utf-8")).hexdigest()
