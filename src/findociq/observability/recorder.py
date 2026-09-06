"""Trace recording primitives with JSONL, memory, and no-op backends."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Protocol

from pydantic import TypeAdapter

from findociq.observability.schema import (
    AttributeValue,
    ObservabilityConfig,
    SpanEvent,
    TraceContext,
)

SPAN_ADAPTER = TypeAdapter(SpanEvent)
LOGGER = logging.getLogger(__name__)


class Clock(Protocol):
    """Monotonic clock boundary used by deterministic tests."""

    def now_ns(self) -> int: ...


class SystemClock:
    """Production monotonic clock."""

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()


class TraceRecorder(Protocol):
    """Sink boundary for immutable span events."""

    def record(self, event: SpanEvent) -> None: ...


class NoOpRecorder:
    """Disabled recorder that intentionally retains no data."""

    def record(self, event: SpanEvent) -> None:
        del event


class InMemoryRecorder:
    """Recorder used by deterministic unit tests."""

    def __init__(self) -> None:
        self.events: list[SpanEvent] = []

    def record(self, event: SpanEvent) -> None:
        self.events.append(event)


class CompositeRecorder:
    """Fan one immutable event out to each configured backend."""

    def __init__(self, *recorders: TraceRecorder) -> None:
        if not recorders:
            raise ValueError("composite recorder requires at least one backend")
        self.recorders = recorders

    def record(self, event: SpanEvent) -> None:
        for recorder in self.recorders:
            try:
                recorder.record(event)
            except Exception:
                LOGGER.warning("observability sink unavailable")


class JsonlRecorder:
    """Append-only local JSONL recorder."""

    def __init__(self, path: str | Path, *, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")
        self._lock = Lock()

    def record(self, event: SpanEvent) -> None:
        line = event.model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class TraceObserver:
    """Times stages and records only structural, content-safe attributes."""

    def __init__(self, recorder: TraceRecorder | None = None, clock: Clock | None = None) -> None:
        self.recorder = recorder or NoOpRecorder()
        self.clock = clock or SystemClock()

    @contextmanager
    def span(
        self,
        context: TraceContext,
        stage: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Iterator[dict[str, AttributeValue]]:
        mutable_attributes = dict(attributes or {})
        started = self.clock.now_ns()
        try:
            yield mutable_attributes
        except Exception as error:
            self._record(
                context,
                stage,
                started,
                "error",
                mutable_attributes,
                type(error).__name__,
            )
            raise
        self._record(context, stage, started, "success", mutable_attributes, None)

    def _record(
        self,
        context: TraceContext,
        stage: str,
        started_ns: int,
        status: str,
        attributes: dict[str, AttributeValue],
        error_type: str | None,
    ) -> None:
        duration_ms = max(0.0, (self.clock.now_ns() - started_ns) / 1_000_000)
        try:
            self.recorder.record(
                SpanEvent(
                **context.model_dump(),
                stage=stage,
                status=status,
                duration_ms=duration_ms,
                attributes=attributes,
                error_type=error_type,
                )
            )
        except Exception:
            LOGGER.warning("observability recording unavailable")


def build_observer(config: ObservabilityConfig, *, reset: bool = False) -> TraceObserver:
    """Construct content-safe JSONL and optional Langfuse trace exporters."""
    if not config.enabled:
        return TraceObserver()
    recorders: list[TraceRecorder] = []
    try:
        recorders.append(JsonlRecorder(config.trace_path, reset=reset))
    except Exception:
        LOGGER.warning("JSONL observability unavailable")
    langfuse = config.langfuse
    if (
        langfuse is not None
        and langfuse.enabled
        and _credential_available(langfuse.public_key_env)
        and _credential_available(langfuse.secret_key_env)
    ):
        from findociq.observability.langfuse import LangfuseOtlpRecorder

        try:
            recorders.append(LangfuseOtlpRecorder(langfuse))
        except Exception:
            LOGGER.warning("Langfuse observability unavailable")
    if not recorders:
        return TraceObserver()
    recorder: TraceRecorder = (
        recorders[0] if len(recorders) == 1 else CompositeRecorder(*recorders)
    )
    return TraceObserver(recorder)


def _credential_available(name: str) -> bool:
    return bool(os.environ.get(name) or os.environ.get(f"{name}_FILE"))


def load_trace_events(path: str | Path) -> tuple[SpanEvent, ...]:
    """Load a JSONL trace artifact through the typed event boundary."""
    events: list[SpanEvent] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(SPAN_ADAPTER.validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid trace event at line {line_number}") from error
    return tuple(events)
