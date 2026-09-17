"""Explicit single-pass/two-pass reasoning orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from findociq.observability.recorder import TraceObserver
from findociq.observability.schema import TraceContext
from findociq.reason.generation import GenerationClient
from findociq.reason.pass1_extract import Pass1Extractor
from findociq.reason.pass2_reason import Pass2Reasoner
from findociq.reason.schema import ReasoningRun
from findociq.reason.single_pass import SinglePassReasoner
from findociq.retrieve.schema import RetrievalHit


@dataclass(frozen=True, slots=True)
class ReasoningPipelineConfig:
    name: str
    passes: Literal[1, 2]

    def __post_init__(self) -> None:
        if not self.name or self.passes not in {1, 2}:
            raise ValueError("reasoning config needs a name and passes of 1 or 2")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReasoningPipelineConfig:
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"name", "passes"}:
            raise ValueError(f"pipeline config must contain exactly name and passes: {source}")
        return cls(**payload)


class ReasoningPipeline:
    def __init__(
        self,
        config: ReasoningPipelineConfig,
        client: GenerationClient,
        observer: TraceObserver | None = None,
    ) -> None:
        self.config = config
        self.observer = observer or TraceObserver()
        self._single = SinglePassReasoner(client, self.observer)
        self._pass1 = Pass1Extractor(client, self.observer)
        self._pass2 = Pass2Reasoner(client, self.observer)

    def run(
        self,
        question: str,
        hits: tuple[RetrievalHit, ...],
        *,
        trace_context: TraceContext | None = None,
    ) -> ReasoningRun:
        context = trace_context or TraceContext.for_query(
            question, operation=f"reasoning:{self.config.name}"
        )
        with self.observer.span(
            context,
            "reasoning.total",
            {"mode": self.config.name, "evidence_count": len(hits)},
        ) as attributes:
            if self.config.passes == 1:
                with self.observer.span(context, "reasoning.single_pass"):
                    answer = self._single.reason(question, hits, trace_context=context)
                attributes["citation_count"] = len(answer.citations)
                return ReasoningRun(mode=self.config.name, question=question, answer=answer)
            with self.observer.span(context, "reasoning.pass1"):
                extraction = self._pass1.extract(question, hits, trace_context=context)
            with self.observer.span(context, "reasoning.pass2"):
                answer = self._pass2.reason(question, extraction, trace_context=context)
            attributes["citation_count"] = len(answer.citations)
            return ReasoningRun(
                mode=self.config.name,
                question=question,
                answer=answer,
                extraction=extraction,
            )
