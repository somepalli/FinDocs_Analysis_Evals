"""Local OpenAI-compatible Gemma generation client.

This module intentionally talks only to a local OpenAI-compatible endpoint.
It does not import or call proprietary model APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from findociq.observability.recorder import TraceObserver
from findociq.observability.schema import TraceContext


class ModelTierConfig(BaseModel):
    """Pinned open-weight model tier served by the local runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["laptop", "single_gpu", "ceiling"]
    source_model_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    quantization: Literal["awq-int4"]
    temperature: Literal[0.0] = 0.0
    seed: int = 17
    max_model_length: int = Field(default=8192, gt=0)
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelTierConfig:
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"model tier config must be a mapping: {source}")
        return cls(**payload)


class GenerationConfig(BaseModel):
    """Typed local generation endpoint and deterministic decoding settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    backend: Literal["ollama", "vllm"] = "vllm"
    temperature: float = 0.0
    seed: int = 17
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: int = Field(default=600, gt=0)
    model_tier: Literal["laptop", "single_gpu", "ceiling"] | None = None
    quantization: Literal["awq-int4"] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> GenerationConfig:
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"generation config must be a mapping: {source}")
        tier_path = payload.pop("tier_config", None)
        if payload.get("backend", "vllm") == "vllm" and tier_path is None:
            raise ValueError("vLLM generation YAML must select a model tier")
        if tier_path is not None:
            if "model_id" in payload or "revision" in payload:
                raise ValueError("generation config cannot mix tier_config with model_id/revision")
            resolved = (source.parent / str(tier_path)).resolve()
            tier = ModelTierConfig.from_yaml(resolved)
            payload.update(
                {
                    "model_id": tier.source_model_id,
                    "revision": tier.revision,
                    "model_tier": tier.name,
                    "quantization": tier.quantization,
                }
            )
        return cls(**payload)

    @model_validator(mode="after")
    def validate_local_endpoint(self) -> GenerationConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
            "vllm",
        }:
            raise ValueError("generation base_url must be a local HTTP endpoint")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return self


class GenerationClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        trace_context: TraceContext | None = None,
        stage: str = "generation",
    ) -> str: ...


class LocalGemmaClient:
    """Minimal deterministic client for a local OpenAI-compatible runtime."""

    def __init__(self, config: GenerationConfig, observer: TraceObserver | None = None) -> None:
        self.config = config
        self.observer = observer or TraceObserver()
        self._revision_validated = False

    def complete(
        self,
        prompt: str,
        *,
        trace_context: TraceContext | None = None,
        stage: str = "generation",
    ) -> str:
        if not prompt.strip():
            raise ValueError("generation prompt must not be blank")
        context = trace_context or TraceContext.for_query(prompt, operation="generation")
        with self.observer.span(
            context,
            stage,
            {
                "backend": self.config.backend,
                "model_id": self.config.model_id,
                "model_revision": self.config.revision,
                "prompt_characters": len(prompt),
                "max_tokens": self.config.max_tokens,
            },
        ) as attributes:
            self._validate_revision()
            endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": self.config.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
            }
            request = Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                    result = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as error:
                raise RuntimeError(f"local Gemma generation failed: {error}") from error
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise RuntimeError(
                    "local Gemma response did not contain message content"
                ) from error
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("local Gemma returned empty message content")
            attributes["response_characters"] = len(content)
        return content

    def _validate_revision(self) -> None:
        if self._revision_validated or self.config.backend != "ollama":
            return
        parsed = urlparse(self.config.base_url)
        endpoint = f"{parsed.scheme}://{parsed.netloc}/api/tags"
        try:
            with urlopen(endpoint, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"local Ollama revision check failed: {error}") from error
        models = payload.get("models", []) if isinstance(payload, dict) else []
        match = next(
            (
                item
                for item in models
                if isinstance(item, dict)
                and item.get("name") in {self.config.model_id, f"{self.config.model_id}:latest"}
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"local Ollama model is not installed: {self.config.model_id}")
        if match.get("digest") != self.config.revision:
            raise RuntimeError(
                f"local Ollama model revision mismatch for {self.config.model_id}: "
                f"expected {self.config.revision}, got {match.get('digest')}"
            )
        self._revision_validated = True


class VllmGemmaClient(LocalGemmaClient):
    """Gemma client bound to a local vLLM OpenAI-compatible endpoint."""

    def __init__(self, config: GenerationConfig, observer: TraceObserver | None = None) -> None:
        if config.backend != "vllm":
            raise ValueError("VllmGemmaClient requires a vLLM generation config")
        super().__init__(config, observer)


class OllamaGemmaClient(LocalGemmaClient):
    """Gemma client bound to Ollama's OpenAI-compatible endpoint."""

    def __init__(self, config: GenerationConfig, observer: TraceObserver | None = None) -> None:
        if config.backend != "ollama":
            raise ValueError("OllamaGemmaClient requires an Ollama generation config")
        super().__init__(config, observer)


def build_generation_client(
    config: GenerationConfig, observer: TraceObserver | None = None
) -> LocalGemmaClient:
    """Select a concrete local serving client from typed configuration."""
    if config.backend == "vllm":
        return VllmGemmaClient(config, observer)
    return OllamaGemmaClient(config, observer)
