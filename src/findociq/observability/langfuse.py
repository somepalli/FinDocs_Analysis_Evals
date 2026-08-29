"""OTLP span exporter for a self-hosted Langfuse deployment."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path
from typing import Any

from findociq.observability.schema import LangfuseOtlpConfig, SpanEvent


class LangfuseOtlpRecorder:
    """Export structural FinDocIQ spans to Langfuse without captured content."""

    def __init__(self, config: LangfuseOtlpConfig) -> None:
        self.config = config
        self._provider: Any | None = None
        self._tracer: Any | None = None

    def _initialize(self) -> None:
        if self._tracer is not None:
            return
        config = self.config
        public_key = _credential(config.public_key_env)
        secret_key = _credential(config.secret_key_env)
        if not public_key or not secret_key:
            raise RuntimeError(
                f"Langfuse requires {config.public_key_env} and {config.secret_key_env}"
            )
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as error:
            raise RuntimeError(
                "Langfuse export requires `uv sync --extra observability`"
            ) from error

        credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        exporter = OTLPSpanExporter(
            endpoint=config.base_url.rstrip("/") + "/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {credentials}"},
            timeout=config.timeout_seconds,
        )
        provider = TracerProvider(resource=Resource.create({"service.name": config.service_name}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        self._provider = provider
        self._tracer = provider.get_tracer("findociq")

    def record(self, event: SpanEvent) -> None:
        self._initialize()
        from opentelemetry.trace import (
            NonRecordingSpan,
            SpanContext,
            Status,
            StatusCode,
            TraceFlags,
            set_span_in_context,
        )

        end_time = time.time_ns()
        start_time = end_time - int(event.duration_ms * 1_000_000)
        trace_id = _nonzero_hash(event.run_id, 32)
        parent_span_id = _nonzero_hash(event.run_id + ":root", 16)
        parent = NonRecordingSpan(
            SpanContext(
                trace_id=trace_id,
                span_id=parent_span_id,
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
        )
        attributes: dict[str, Any] = {
            "findociq.run_id": event.run_id,
            "findociq.operation": event.operation,
            "findociq.query_sha256": event.query_sha256,
            "findociq.status": event.status,
            **{f"findociq.{key}": value for key, value in event.attributes.items()},
        }
        for key, value in (
            ("findociq.question_id", event.question_id),
            ("findociq.config_hash", event.config_hash),
            ("findociq.dataset_sha256", event.dataset_sha256),
            ("findociq.prompt_template_id", event.prompt_template_id),
            ("findociq.prompt_version", event.prompt_version),
            ("findociq.prompt_sha256", event.prompt_sha256),
            ("findociq.error_type", event.error_type),
        ):
            if value is not None:
                attributes[key] = value
        attributes = {key: value for key, value in attributes.items() if value is not None}
        span = self._tracer.start_span(
            event.stage,
            context=set_span_in_context(parent),
            start_time=start_time,
            attributes=attributes,
        )
        if event.status == "error":
            span.set_status(Status(StatusCode.ERROR, event.error_type or "pipeline error"))
        span.end(end_time=end_time)

    def shutdown(self) -> None:
        if self._provider is not None:
            self._provider.shutdown()


def _nonzero_hash(value: str, hex_characters: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:hex_characters], 16) or 1


def _credential(name: str) -> str | None:
    path = os.environ.get(f"{name}_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return os.environ.get(name)
