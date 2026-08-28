from pathlib import Path

import pytest

from findociq.observability.aggregate import aggregate_traces, write_observability_report
from findociq.observability.langfuse import LangfuseOtlpRecorder
from findociq.observability.recorder import (
    CompositeRecorder,
    InMemoryRecorder,
    JsonlRecorder,
    TraceObserver,
)
from findociq.observability.schema import LangfuseOtlpConfig, ObservabilityConfig, TraceContext
from findociq.retrieve.pipeline import RetrievalPipeline, RetrievalStrategyConfig
from tests.test_retrieval import FakeEmbedder, FakeStore

ROOT = Path(__file__).parents[1]


class StepClock:
    def __init__(self, *values: int) -> None:
        self.values = list(values)

    def now_ns(self) -> int:
        return self.values.pop(0)


def context() -> TraceContext:
    return TraceContext.for_query(
        "What was revenue?",
        operation="test",
        question_id="q1",
        config_hash="a" * 64,
        dataset_sha256="b" * 64,
    )


def strategy() -> RetrievalStrategyConfig:
    return RetrievalStrategyConfig(
        name="naive",
        mode="dense",
        retrieve_top_k=3,
        rerank_top_k=None,
        prefetch_top_k=5,
        rrf_k=60,
    )


def test_observability_config_is_typed_and_content_safe() -> None:
    config = ObservabilityConfig.from_yaml(ROOT / "configs/observability/local.yaml")
    assert config.enabled is True
    assert config.capture_content is False
    with pytest.raises(ValueError):
        ObservabilityConfig(enabled=True, capture_content=True)  # type: ignore[arg-type]
    trace_context = context()
    serialized = trace_context.model_dump_json()
    assert "What was revenue?" not in serialized
    assert len(trace_context.query_sha256) == 64
    assert trace_context.run_id == context().run_id
    langfuse = ObservabilityConfig.from_yaml(ROOT / "configs/observability/langfuse.yaml")
    assert langfuse.langfuse is not None
    assert langfuse.langfuse.enabled is True
    with pytest.raises(ValueError, match="self-hosted"):
        LangfuseOtlpConfig(enabled=True, base_url="https://cloud.langfuse.com")


def test_langfuse_configuration_accepts_docker_desktop_host_gateway() -> None:
    config = LangfuseOtlpConfig(
        enabled=True,
        base_url="http://host.docker.internal:3000",
    )
    assert config.base_url == "http://host.docker.internal:3000"


def test_observer_records_success_and_error_without_exception_text() -> None:
    recorder = InMemoryRecorder()
    observer = TraceObserver(recorder, StepClock(0, 2_000_000, 3_000_000, 8_000_000))
    with observer.span(context(), "success", {"count": 2}):
        pass
    with (
        pytest.raises(ValueError, match="secret detail"),
        observer.span(context(), "failure"),
    ):
        raise ValueError("secret detail")
    assert recorder.events[0].duration_ms == 2.0
    assert recorder.events[0].status == "success"
    assert recorder.events[1].status == "error"
    assert recorder.events[1].error_type == "ValueError"
    assert "secret detail" not in recorder.events[1].model_dump_json()


def test_retrieval_traces_cache_and_preserves_identical_results() -> None:
    recorder = InMemoryRecorder()
    clock = StepClock(*range(0, 20_000_000, 1_000_000))
    observed = RetrievalPipeline(
        strategy(), FakeEmbedder(), FakeStore(), observer=TraceObserver(recorder, clock)
    )
    baseline = RetrievalPipeline(strategy(), FakeEmbedder(), FakeStore())
    first = observed.retrieve("revenue", trace_context=context())
    second = observed.retrieve("revenue", trace_context=context())
    assert first == baseline.retrieve("revenue")
    assert second is first
    totals = [event for event in recorder.events if event.stage == "retrieval.total"]
    assert [event.attributes["cache_hit"] for event in totals] == [False, True]
    assert all(event.attributes["hit_count"] == 3 for event in totals)


def test_aggregation_keeps_operational_metrics_separate(tmp_path: Path) -> None:
    recorder = InMemoryRecorder()
    observer = TraceObserver(recorder, StepClock(0, 1_000_000, 2_000_000, 7_000_000))
    with observer.span(
        context(), "retrieval.total", {"cache_hit": False, "hit_count": 3}
    ):
        pass
    with observer.span(
        TraceContext.for_query("empty", operation="retrieval:test"),
        "retrieval.total",
        {"cache_hit": True, "hit_count": 0},
    ):
        pass
    summary = aggregate_traces(recorder.events)
    assert summary.run_count == 2
    assert summary.cache_hit_rate == 0.5
    assert summary.non_empty_retrieval_rate == 0.5
    assert summary.stages[0].p50_ms == 1.0
    assert summary.stages[0].p95_ms == 5.0
    json_path, markdown_path = write_observability_report(summary, tmp_path)
    assert json_path.exists()
    assert "Operational metrics" in markdown_path.read_text(encoding="utf-8")


def test_jsonl_recorder_resets_and_writes_typed_events(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text("stale\n", encoding="utf-8")
    recorder = JsonlRecorder(path, reset=True)
    observer = TraceObserver(recorder, StepClock(0, 1_000_000))
    with observer.span(context(), "stage"):
        pass
    content = path.read_text(encoding="utf-8")
    assert "stale" not in content
    assert '"stage":"stage"' in content


def test_composite_recorder_fans_out_identical_safe_events() -> None:
    first, second = InMemoryRecorder(), InMemoryRecorder()
    observer = TraceObserver(CompositeRecorder(first, second), StepClock(0, 1_000_000))
    with observer.span(context(), "generation.single_pass", {"model_id": "gemma"}):
        pass
    assert first.events == second.events
    assert "What was revenue?" not in first.events[0].model_dump_json()


def test_langfuse_export_requires_environment_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    recorder = LangfuseOtlpRecorder(LangfuseOtlpConfig(enabled=True))
    with (
        pytest.raises(RuntimeError, match="LANGFUSE_PUBLIC_KEY"),
        TraceObserver(recorder, StepClock(0, 1_000_000)).span(context(), "stage"),
    ):
        pass
