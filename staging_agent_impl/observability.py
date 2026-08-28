"""Content-safe, FunderMatch-local agent telemetry."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from fundermatch.orchestration.schema import GraphStatus, WorkerName

FORBIDDEN_TRACE_KEYS = frozenset(
    {
        "borrower_name",
        "financial_value",
        "pdf",
        "text",
        "prompt",
        "answer",
        "messages",
        "checkpoint_state",
        "exception",
        "credential",
        "token",
    }
)


class AgentSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    span_id: str = Field(default_factory=lambda: uuid4().hex)
    application_id: str
    job_id: str | None = None
    run_id: str
    thread_id: str
    checkpoint_id: str | None = None
    worker: WorkerName | None = None
    attempt: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    status: GraphStatus
    model_revision: str | None = None
    config_hash: str | None = None
    document_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    guardrail_code: str | None = None
    error_code: str | None = None
    event: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentSpanRecorder(Protocol):
    async def record(self, event: AgentSpan) -> None: ...


class NoopAgentSpanRecorder:
    async def record(self, event: AgentSpan) -> None:
        del event


class JsonlAgentSpanRecorder:
    """Local fallback whose schema cannot represent borrower content."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    async def record(self, event: AgentSpan) -> None:
        payload = event.model_dump_json()
        assert_trace_is_content_safe(json.loads(payload))
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(payload + "\n")


class CompositeAgentSpanRecorder:
    """Telemetry is best-effort and can never stop business execution."""

    def __init__(self, *recorders: AgentSpanRecorder) -> None:
        self.recorders = recorders

    async def record(self, event: AgentSpan) -> None:
        await asyncio.gather(
            *(self._safe_record(recorder, event) for recorder in self.recorders)
        )

    @staticmethod
    async def _safe_record(recorder: AgentSpanRecorder, event: AgentSpan) -> None:
        try:
            await recorder.record(event)
        except Exception:
            return


class OtlpAgentSpanRecorder:
    """Lazy OpenTelemetry adapter for the existing self-hosted Langfuse endpoint."""

    def __init__(self, *, endpoint: str, headers: dict[str, str] | None = None) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError as error:
            raise RuntimeError("OpenTelemetry OTLP dependencies are not installed") from error
        provider = TracerProvider(
            resource=Resource.create({"service.name": "fundermatch-agents"})
        )
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers or {}))
        )
        self._tracer = provider.get_tracer("fundermatch.orchestration")
        self._trace = trace

    async def record(self, event: AgentSpan) -> None:
        attributes = event.model_dump(mode="json", exclude_none=True)
        attributes.pop("occurred_at", None)
        with self._tracer.start_as_current_span(
            f"agent.{event.worker.value if event.worker else 'supervisor'}"
        ) as span:
            for key, value in attributes.items():
                span.set_attribute(f"fundermatch.{key}", value)


def assert_trace_is_content_safe(payload: dict[str, object]) -> None:
    forbidden = FORBIDDEN_TRACE_KEYS.intersection(key.casefold() for key in payload)
    if forbidden:
        raise ValueError(f"forbidden trace fields: {sorted(forbidden)}")
