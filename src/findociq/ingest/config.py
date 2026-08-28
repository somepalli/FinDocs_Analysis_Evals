"""Typed YAML configuration for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.resources import files
from json import dumps
from pathlib import Path
from typing import Any

import yaml

from findociq.ingest.chunker import ChunkerConfig
from findociq.ingest.docling_parser import ParserConfig
from findociq.ingest.gpu_lease import GpuLeaseConfig
from findociq.ingest.router import RouterConfig
from findociq.ingest.vlm_fallback import VisionConfig


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    router: RouterConfig
    parser: ParserConfig
    vision: VisionConfig
    chunker: ChunkerConfig
    gpu_lease: GpuLeaseConfig
    observability_config: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> IngestionConfig:
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"ingestion config must be a mapping: {source}")
        cls._require_sections(payload, source)
        return cls(
            router=RouterConfig(**cls._mapping(payload["router"], "router")),
            parser=ParserConfig(**cls._mapping(payload["parser"], "parser")),
            vision=VisionConfig(**cls._mapping(payload["vision"], "vision")),
            chunker=ChunkerConfig(**cls._mapping(payload["chunker"], "chunker")),
            gpu_lease=GpuLeaseConfig(**cls._mapping(payload["gpu_lease"], "gpu_lease")),
            observability_config=(source.parent / payload["observability_config"]).resolve(),
        )

    @property
    def config_hash(self) -> str:
        payload = {
            "router": asdict(self.router),
            "parser": asdict(self.parser),
            "vision": self.vision.model_dump(mode="json"),
            "chunker": asdict(self.chunker),
            "gpu_lease": self.gpu_lease.model_dump(mode="json"),
            "observability_config_sha256": sha256(
                self.observability_config.read_bytes()
            ).hexdigest(),
            "vision_prompt_sha256": sha256(
                files("findociq.ingest.prompts")
                .joinpath("vision_extract.txt")
                .read_bytes()
            ).hexdigest(),
        }
        serialized = dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _require_sections(payload: dict[str, Any], source: Path) -> None:
        expected = {
            "router",
            "parser",
            "vision",
            "chunker",
            "gpu_lease",
            "observability_config",
        }
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"invalid ingestion config sections in {source}; missing={missing}, extra={extra}"
            )

    @staticmethod
    def _mapping(value: Any, section: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"ingestion config section '{section}' must be a mapping")
        return value
