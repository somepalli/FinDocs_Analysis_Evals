"""Application service that composes retrieval and grounded reasoning."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from findociq.observability.recorder import TraceObserver, build_observer
from findociq.observability.schema import ObservabilityConfig, TraceContext
from findociq.reason.generation import GenerationConfig, build_generation_client
from findociq.reason.pipeline import ReasoningPipeline, ReasoningPipelineConfig
from findociq.reason.schema import ReasoningRun
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
    retrieval_config: Path
    generation_config: Path
    single_pass_config: Path
    two_pass_config: Path
    observability_config: Path
    default_mode: ReasoningMode = "two_pass"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ApiConfig:
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"API config must be a mapping: {source}")
        for key in (
            "index_config",
            "retrieval_config",
            "generation_config",
            "single_pass_config",
            "two_pass_config",
            "observability_config",
        ):
            payload[key] = (source.parent / payload[key]).resolve()
        return cls(**payload)


class FinDocIQService:
    """Framework-independent query service used by CLI and HTTP boundaries."""

    def __init__(
        self,
        retrieval: RetrievalPipeline,
        reasoning: dict[ReasoningMode, ReasoningPipeline],
        observer: TraceObserver | None = None,
        default_mode: ReasoningMode = "two_pass",
    ) -> None:
        self.retrieval = retrieval
        self.reasoning = reasoning
        self.observer = observer or TraceObserver()
        self.default_mode = default_mode

    def query(
        self,
        question: str,
        *,
        mode: ReasoningMode,
        question_id: str | None = None,
        document_ids: tuple[str, ...] = (),
    ) -> ReasoningRun:
        if mode not in self.reasoning:
            raise ValueError(f"unsupported reasoning mode: {mode}")
        context = TraceContext.for_query(
            question,
            operation=f"api:query:{mode}",
            question_id=question_id,
        )
        with self.observer.span(context, "api.query", {"mode": mode}):
            hits = self.retrieval.retrieve(
                question, trace_context=context, document_ids=document_ids
            )
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
    return FinDocIQService(retrieval, pipelines, observer, config.default_mode)
