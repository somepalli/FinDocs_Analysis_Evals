"""Typed, hashed production guardrail policy loaded from reviewed YAML."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResourcePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_pdf_bytes: int = Field(gt=0)
    max_batch_bytes: int = Field(gt=0)
    max_pages: int = Field(gt=0)
    max_batch_documents: int = Field(gt=0)
    parse_timeout_seconds: int = Field(gt=0)
    max_concurrent_batches: int = Field(gt=0)

    @model_validator(mode="after")
    def batch_can_hold_one_document(self) -> ResourcePolicy:
        if self.max_batch_bytes < self.max_pdf_bytes:
            raise ValueError("max_batch_bytes must be at least max_pdf_bytes")
        return self


class ServiceAuthPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    issuer: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    maximum_token_age_seconds: int = Field(gt=0, le=300)
    required_correlation_header: str = Field(min_length=1)


class DlpPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    replacement: str = "[REDACTED:{kind}]"
    protected_kinds: tuple[str, ...] = Field(min_length=1)
    instruction_patterns: tuple[str, ...] = Field(min_length=1)


class RetentionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    terminal_days: int = Field(gt=0)
    quarantine_days: int = Field(gt=0)


class WorkerExecutionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    permitted_tools: tuple[str, ...]
    max_calls: int = Field(gt=0)
    call_timeout_seconds: int = Field(gt=0)
    worker_deadline_seconds: int = Field(gt=0)
    max_attempts: int = Field(gt=0)
    backoff_seconds: float = Field(ge=0)
    side_effects: tuple[str, ...] = ()


class ApiLimitPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: int = Field(gt=0)
    window_seconds: int = Field(gt=0)
    max_concurrent: int = Field(gt=0)
    concurrency_lease_seconds: int = Field(gt=0)


class PrecedentPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_validity_days: int = Field(gt=0)
    require_human_decision: bool = True


class BrandPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    required_notice: str = Field(min_length=1)
    prohibited_terms: tuple[str, ...] = Field(min_length=1)


class ProductionGuardrailPolicy(BaseModel):
    """Single policy identity propagated through all production receipts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    permitted_metric_ids: dict[str, str] = Field(min_length=1)
    resources: ResourcePolicy
    service_auth: ServiceAuthPolicy
    dlp: DlpPolicy
    retention: RetentionPolicy
    workers: dict[str, WorkerExecutionPolicy] = Field(min_length=1)
    api_limits: dict[str, ApiLimitPolicy] = Field(min_length=1)
    precedent: PrecedentPolicy
    brand: BrandPolicy
    policy_hash: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProductionGuardrailPolicy:
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"guardrail policy must be a mapping: {source}")
        supplied_hash = payload.pop("policy_hash", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        calculated = sha256(canonical.encode()).hexdigest()
        if supplied_hash and supplied_hash != calculated:
            raise ValueError("guardrail policy hash does not match its content")
        return cls(**payload, policy_hash=calculated)

    def extraction_question(self, metric_id: str) -> str:
        try:
            return self.permitted_metric_ids[metric_id]
        except KeyError as error:
            raise ValueError(f"metric is not allowed by production policy: {metric_id}") from error
