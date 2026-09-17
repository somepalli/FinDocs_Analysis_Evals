"""Application service that composes retrieval and grounded reasoning."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from findociq.ingest.config import IngestionConfig
from findociq.ingest.gpu_lease import GpuLeaseConfig, VllmGpuLease
from findociq.observability.recorder import TraceObserver, build_observer
from findociq.observability.schema import ObservabilityConfig, TraceContext
from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, EvidencePolicy
from findociq.reason.generation import GenerationConfig, build_generation_client
from findociq.reason.pipeline import ReasoningPipeline, ReasoningPipelineConfig
from findociq.reason.schema import Pass1Extraction, ReasoningRun
from findociq.retrieve.pipeline import (
    RetrievalPipeline,
    RetrievalRuntimeConfig,
    build_local_pipeline,
)

ReasoningMode = Literal["single_pass", "two_pass"]


class ApiConfig(BaseModel):
    """Typed paths for the thin local API composition root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index_config: Path
    ingestion_config: Path
    retrieval_config: Path
    generation_config: Path
    single_pass_config: Path
    two_pass_config: Path
    observability_config: Path
    default_mode: ReasoningMode = "two_pass"
    evidence_config: Path = Path("configs/evidence/borrower.yaml")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ApiConfig:
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"API config must be a mapping: {source}")
        for key in (
            "index_config",
            "ingestion_config",
            "retrieval_config",
            "generation_config",
            "single_pass_config",
            "two_pass_config",
            "observability_config",
        ):
            payload[key] = (source.parent / payload[key]).resolve()
        payload["evidence_config"] = (
            source.parent / payload.get("evidence_config", "../evidence/borrower.yaml")
        ).resolve()
        return cls(**payload)


class FinDocIQService:
    """Framework-independent query service used by CLI and HTTP boundaries."""

    def __init__(
        self,
        retrieval: RetrievalPipeline,
        reasoning: dict[ReasoningMode, ReasoningPipeline],
        observer: TraceObserver | None = None,
        default_mode: ReasoningMode = "two_pass",
        gpu_lease: VllmGpuLease | None = None,
        evidence_policy: EvidencePolicy | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.reasoning = reasoning
        self.observer = observer or TraceObserver()
        self.default_mode = default_mode
        self.gpu_lease = gpu_lease or VllmGpuLease(GpuLeaseConfig())
        self.evidence_policy = evidence_policy

    def assess_metrics(
        self,
        metric_ids: tuple[str, ...],
        document_ids: tuple[str, ...],
        *,
        application_id: str,
        command_id: str,
        interpretations=(),
        interpretation_command_id=None,
        assessment_date=None,
    ):
        from findociq.reason.classification import classify_sections
        from findociq.reason.interpretations import interpretation_options

        policy = self.evidence_policy
        if policy is None:
            raise RuntimeError("evidence_policy_unavailable")
        if any(
            metric not in policy.metrics and metric not in policy.operands for metric in metric_ids
        ):
            raise EvidenceInsufficient("metric_not_allowed")
        context = TraceContext.for_query(
            "Evidence readiness",
            operation="evidence.assessment",
            question_id=command_id,
            config_hash=policy.policy_hash,
        )
        with self.observer.span(
            context,
            "evidence.assessment",
            {"document_count": len(document_ids), "metric_count": len(metric_ids)},
        ):
            chunks = self.retrieval.store.scoped_chunks(
                document_ids, application_id, max_chunks=policy.max_chunks
            )
            if any(
                link.document_id not in document_ids
                or not any(
                    c.chunk_id == link.target_chunk_id
                    and all(p.document_id == link.document_id for p in c.provenance)
                    for c in chunks
                )
                for link in interpretations
            ):
                raise PermissionError("evidence_scope_violation")
            return (
                BorrowerEvidenceGate(
                    policy,
                    interpretations=interpretations,
                    interpretation_command_id=interpretation_command_id,
                    assessment_date=assessment_date,
                    application_id=application_id,
                ).assess(metric_ids, chunks),
                classify_sections(chunks, policy.classification_policy),
                interpretation_options(chunks, policy),
            )

    def extract_metrics(
        self,
        question: str,
        metric_ids: tuple[str, ...],
        document_ids: tuple[str, ...],
        *,
        application_id: str | None = None,
        question_id: str | None = None,
    ) -> Pass1Extraction:
        if self.evidence_policy is None:
            raise RuntimeError("evidence policy unavailable")
        context = TraceContext.for_query(
            question,
            operation="evidence.selection",
            question_id=question_id,
            config_hash=self.evidence_policy.policy_hash,
        )
        with self.observer.span(
            context,
            "evidence.selection",
            {"document_count": len(document_ids), "metric_count": len(metric_ids)},
        ):
            try:
                chunks = self.retrieval.store.scoped_chunks(
                    document_ids, application_id, max_chunks=self.evidence_policy.max_chunks
                )
            except ValueError as error:
                code = str(error)
                if code not in {
                    "evidence_document_scope_missing",
                    "evidence_inventory_limit",
                    "evidence_document_missing",
                }:
                    code = "evidence_inventory_invalid"
                raise EvidenceInsufficient(code) from error
            gate = BorrowerEvidenceGate(self.evidence_policy, application_id=application_id)
            figures = tuple(gate.extract(metric, chunks) for metric in metric_ids)
            return Pass1Extraction(question=question, figures=figures)

    def query(
        self,
        question: str,
        *,
        mode: ReasoningMode,
        question_id: str | None = None,
        document_ids: tuple[str, ...] = (),
        application_id: str | None = None,
    ) -> ReasoningRun:
        if mode not in self.reasoning:
            raise ValueError(f"unsupported reasoning mode: {mode}")
        context = TraceContext.for_query(
            question,
            operation=f"api:query:{mode}",
            question_id=question_id,
        )
        with self.gpu_lease.operation(), self.observer.span(context, "api.query", {"mode": mode}):
            with self.gpu_lease.ingestion_batch():
                try:
                    hits = self.retrieval.retrieve(
                        question,
                        trace_context=context,
                        document_ids=document_ids,
                        application_id=application_id,
                    )
                finally:
                    if self.gpu_lease.config.enabled:
                        self.retrieval.release_models()
            return self.reasoning[mode].run(question, hits, trace_context=context)


def build_service(config: ApiConfig) -> FinDocIQService:
    """Build the local open-weight pipeline from reviewed YAML configuration."""
    observability = ObservabilityConfig.from_yaml(config.observability_config)
    observer = build_observer(observability)
    runtime = RetrievalRuntimeConfig.from_yaml(config.index_config, config.retrieval_config)
    retrieval = build_local_pipeline(runtime, observer)
    generation = GenerationConfig.from_yaml(config.generation_config)
    client = build_generation_client(generation, observer)
    pipelines: dict[ReasoningMode, ReasoningPipeline] = {
        "single_pass": ReasoningPipeline(
            ReasoningPipelineConfig.from_yaml(config.single_pass_config), client, observer
        ),
        "two_pass": ReasoningPipeline(
            ReasoningPipelineConfig.from_yaml(config.two_pass_config), client, observer
        ),
    }
    lease = VllmGpuLease(IngestionConfig.from_yaml(config.ingestion_config).gpu_lease)
    return FinDocIQService(
        retrieval,
        pipelines,
        observer,
        config.default_mode,
        gpu_lease=lease,
        evidence_policy=EvidencePolicy.load(config.evidence_config),
    )
