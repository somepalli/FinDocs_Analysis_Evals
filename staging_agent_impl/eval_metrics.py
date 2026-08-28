"""Separate retrieval and CPU/GPU performance reporting for release gates."""

from __future__ import annotations

import math
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    expected_precedent_ids: tuple[str, ...] = Field(min_length=1)
    ranked_precedent_ids: tuple[str, ...]
    forbidden_funder_leakage: bool = False


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=1)
    eligible_only_recall_at_3: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_3: float = Field(ge=0, le=1)


class PerformanceObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: Literal["cpu", "gpu"]
    stage: str
    latency_ms: float = Field(ge=0)


class PerformanceMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: Literal["cpu", "gpu"]
    stage: str
    n: int = Field(ge=1)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


def retrieval_metrics(observations: tuple[RetrievalObservation, ...]) -> RetrievalMetrics:
    if not observations:
        raise ValueError("retrieval metrics require at least one observation")
    if any(item.forbidden_funder_leakage for item in observations):
        raise ValueError("eligible-only retrieval contained a forbidden funder")
    recalls = []
    reciprocal_ranks = []
    ndcgs = []
    for item in observations:
        expected = set(item.expected_precedent_ids)
        top_three = item.ranked_precedent_ids[:3]
        recalls.append(len(expected.intersection(top_three)) / len(expected))
        rank = next(
            (index for index, case_id in enumerate(item.ranked_precedent_ids, start=1) if case_id in expected),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        dcg = sum(
            1.0 / math.log2(index + 1)
            for index, case_id in enumerate(top_three, start=1)
            if case_id in expected
        )
        ideal = sum(
            1.0 / math.log2(index + 1)
            for index in range(1, min(3, len(expected)) + 1)
        )
        ndcgs.append(dcg / ideal)
    return RetrievalMetrics(
        n=len(observations),
        eligible_only_recall_at_3=sum(recalls) / len(recalls),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
        ndcg_at_3=sum(ndcgs) / len(ndcgs),
    )


def performance_metrics(
    observations: tuple[PerformanceObservation, ...],
) -> tuple[PerformanceMetrics, ...]:
    groups: dict[tuple[str, str], list[float]] = {}
    for item in observations:
        groups.setdefault((item.environment, item.stage), []).append(item.latency_ms)
    results = []
    for (environment, stage), values in sorted(groups.items()):
        ordered = sorted(values)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        results.append(
            PerformanceMetrics(
                environment=environment,
                stage=stage,
                n=len(ordered),
                p50_ms=median(ordered),
                p95_ms=ordered[p95_index],
            )
        )
    return tuple(results)
