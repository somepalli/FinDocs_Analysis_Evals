from pathlib import Path

import pytest

from findociq.ingest.config import IngestionConfig

ROOT = Path(__file__).parents[1]


def test_ingestion_config_loads_into_nested_dataclasses() -> None:
    config = IngestionConfig.from_yaml(ROOT / "configs/ingestion/default.yaml")
    assert config.router.min_digital_characters == 40
    assert config.parser.prefer_docling is True
    assert config.parser.accelerator_device == "cuda"
    assert config.parser.ocr_backend == "torch"
    assert config.vision.backend == "vllm"
    assert config.vision.temperature == 0.0
    assert config.observability_config.name == "langfuse.yaml"
    assert config.chunker.max_text_characters == 1800
    assert config.gpu_lease.enabled is True
    assert config.gpu_lease.sleep_level == 1
    assert len(config.config_hash) == 64
    assert config.config_hash == config.config_hash


def test_ingestion_config_rejects_unknown_sections(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("router: {}\nparser: {}\nchunker: {}\nunknown: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra=\\['unknown'\\]"):
        IngestionConfig.from_yaml(path)
