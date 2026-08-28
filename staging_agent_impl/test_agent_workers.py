from __future__ import annotations

import asyncio
from collections import Counter
from decimal import Decimal

from langgraph.checkpoint.memory import InMemorySaver

from fundermatch.clients.findociq_contract import (
    BoundingBox,
    ExtractResponse,
    ExtractedFigure,
    IngestBatchResponse,
    IngestDocumentResponse,
    SourceCitation,
)
from fundermatch.intake import IntakeMetadata
from fundermatch.matching.schema import RuleGatedRetrievalResult
from fundermatch.orchestration.graph import ApplicationMemoryGraph
from fundermatch.orchestration.guardrails import GuardrailWorker
from fundermatch.orchestration.lifecycle import InMemoryLifecycleStore
from fundermatch.orchestration.schema import GraphStatus, WorkerName
from fundermatch.orchestration.workers import WorkerDependencies, fixed_workers
from fundermatch.orchestration.workspace import ApplicationWorkspace
from fundermatch.rules.engine import EligibilityEngine
from fundermatch.rules.schema import FunderPolicy
from fundermatch.workflow.repository import InMemoryWorkflowRepository
from fundermatch.workflow.schema import ActorClaims, ActorRole, WorkflowState
from fundermatch.workflow.service import WorkflowService


class FakeFinDocIQ:
    def __init__(self) -> None:
        self.extract_calls: Counter[str] = Counter()

    async def ingest_batch(self, request):  # type: ignore[no-untyped-def]
        return IngestBatchResponse(
            contract_version="1.0",
            documents=tuple(
                IngestDocumentResponse(
                    contract_version="1.0",
                    document_id=f"doc-{index}",
                    filename=item.filename,
                    sha256=item.sha256,
                    page_count=2,
                    chunk_count=4,
                    chunk_ids=(f"chunk-{index}",),
                    config_hash="a" * 64,
                )
                for index, item in enumerate(request.documents, start=1)
            ),
        )

    async def extract(self, request):  # type: ignore[no-untyped-def]
        metric = request.question_id.rsplit(":", 1)[-1]
        self.extract_calls[metric] += 1
        if metric == "dscr" and self.extract_calls[metric] == 1:
            raise RuntimeError("injected")
        values = {
            "annual_revenue_crore": ("Revenue", "100", "INR crore"),
            "ebitda_margin_pct": ("EBITDA margin", "12.5", "%"),
            "dscr": ("DSCR", "1.5", "x"),
        }
        label, value, unit = values[metric]
        return ExtractResponse(
            contract_version="1.0",
            question=request.question,
            figures=(
                ExtractedFigure(
                    label=label,
                    value=value,
                    unit=unit,
                    period="FY2026",
                    citation=SourceCitation(
                        document_id="doc-1",
                        page_number=1,
                        bbox=BoundingBox(x0=1, y0=2, x1=3, y1=4),
                    ),
                ),
            ),
        )


class FakeRetriever:
    def retrieve(self, application, policies):  # type: ignore[no-untyped-def]
        return RuleGatedRetrievalResult(
            application_id=application.application_id,
            eligibility=EligibilityEngine().evaluate_all(application, policies),
            matches=(),
        )


def test_fixed_worker_flow_resumes_financial_substeps_without_repeating_them(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> None:
        workspace = ApplicationWorkspace(tmp_path)
        metadata = IntakeMetadata(
            application_id="APP-AGENT-001",
            borrower_name="Synthetic Borrower",
            industry="Manufacturing",
            region="North",
            requested_amount_crore=Decimal("12"),
            debt_to_ebitda=Decimal("1.2"),
            collateral_cover=Decimal("1.2"),
            years_operating=10,
            employee_count=200,
            finance_context="Synthetic finance context",
            operations_context="Synthetic operations context",
        )
        references = workspace.stage(
            metadata,
            (("borrower.pdf", b"%PDF-1.7 synthetic"),),
            max_file_bytes=1024,
            max_batch_bytes=2048,
        )
        policy = FunderPolicy(
            funder_id="test-funder",
            display_name="Test Funder",
            allowed_industries=frozenset({"Manufacturing"}),
            allowed_regions=frozenset({"North"}),
            min_requested_amount_crore=Decimal("1"),
            max_requested_amount_crore=Decimal("20"),
            min_dscr=Decimal("1"),
            max_debt_to_ebitda=Decimal("3"),
            min_collateral_cover=Decimal("1"),
            min_years_operating=3,
        )
        findociq = FakeFinDocIQ()
        workflow = WorkflowService(InMemoryWorkflowRepository())
        dependencies = WorkerDependencies(
            workspace=workspace,
            findociq=findociq,  # type: ignore[arg-type]
            workflow=workflow,
            retriever=FakeRetriever(),  # type: ignore[arg-type]
            policies=(policy,),
            actor=ActorClaims(
                actor_id="agent-pipeline",
                display_name="Agent Pipeline",
                roles=frozenset({ActorRole.PIPELINE}),
            ),
        )
        workers = fixed_workers(dependencies, GuardrailWorker(workspace))
        assert tuple(worker.name for worker in workers) == tuple(WorkerName)
        graph = ApplicationMemoryGraph(
            workers=workers,  # type: ignore[arg-type]
            checkpointer=InMemorySaver(),
            lifecycle=InMemoryLifecycleStore(),
        )

        failed = await graph.start(
            metadata.application_id,
            job_id="intake-agent-001",
            input_references=references,
        )
        assert failed.status == GraphStatus.FAILED_RETRYABLE
        assert failed.current_node == WorkerName.FINANCIAL_ANALYSIS
        assert workspace.exists(metadata.application_id, "financial/annual_revenue_crore")
        assert workspace.exists(metadata.application_id, "financial/ebitda_margin_pct")

        resumed = await graph.resume(metadata.application_id)
        assert resumed.status == GraphStatus.WAITING_FOR_REVIEW
        assert resumed.completed_workers == tuple(WorkerName)
        assert findociq.extract_calls == {
            "annual_revenue_crore": 1,
            "ebitda_margin_pct": 1,
            "dscr": 2,
        }
        record = await workflow.get(metadata.application_id)
        assert record.state == WorkflowState.AWAITING_HUMAN
        assert resumed.human_review is not None
        assert all(item.passed for item in resumed.guardrails)

    asyncio.run(scenario())
