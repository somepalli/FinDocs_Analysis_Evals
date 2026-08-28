import pytest

from fundermatch.evals.metrics import (
    PerformanceObservation,
    RetrievalObservation,
    performance_metrics,
    retrieval_metrics,
)


def test_retrieval_metrics_are_separate_eligible_only_and_report_n() -> None:
    metrics = retrieval_metrics(
        (
            RetrievalObservation(
                case_id="one",
                expected_precedent_ids=("a",),
                ranked_precedent_ids=("a", "b"),
            ),
            RetrievalObservation(
                case_id="two",
                expected_precedent_ids=("c",),
                ranked_precedent_ids=("x", "c"),
            ),
        )
    )
    assert metrics.n == 2
    assert metrics.eligible_only_recall_at_3 == 1.0
    assert metrics.mrr == 0.75
    assert metrics.ndcg_at_3 < 1.0
    assert "accuracy" not in metrics.model_fields


def test_retrieval_report_fails_on_forbidden_funder_leakage() -> None:
    with pytest.raises(ValueError, match="forbidden funder"):
        retrieval_metrics(
            (
                RetrievalObservation(
                    case_id="bad",
                    expected_precedent_ids=("a",),
                    ranked_precedent_ids=("a",),
                    forbidden_funder_leakage=True,
                ),
            )
        )


def test_performance_metrics_never_mix_cpu_and_gpu_labels() -> None:
    rows = performance_metrics(
        (
            PerformanceObservation(environment="cpu", stage="rerank", latency_ms=100),
            PerformanceObservation(environment="cpu", stage="rerank", latency_ms=200),
            PerformanceObservation(environment="gpu", stage="rerank", latency_ms=10),
            PerformanceObservation(environment="gpu", stage="rerank", latency_ms=20),
        )
    )
    assert [(item.environment, item.n) for item in rows] == [("cpu", 2), ("gpu", 2)]
