import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from findociq.api.ingestion import validate_parser_security
from findociq.ingest.config import IngestionConfig
from findociq.ingest.docling_parser import rapidocr_parameters


def test_cuda_ocr_requests_actual_torch_cuda_backend(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    assert rapidocr_parameters("cuda") == {
        "EngineConfig.torch.use_cuda": True,
        "EngineConfig.torch.cuda_ep_cfg.device_id": 0,
    }


def test_explicit_cuda_never_silently_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    with pytest.raises(RuntimeError, match="ocr_cuda_unavailable"):
        rapidocr_parameters("cuda")
    assert rapidocr_parameters("cpu") == {}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_production_allows_local_ocr_but_blocks_unredacted_vision(device):
    config = IngestionConfig.from_yaml("configs/ingestion/docker.yaml")
    config = replace(config, parser=replace(config.parser, accelerator_device=device))
    validate_parser_security(config, production_enabled=True)
    unsafe = replace(config, vision=config.vision.model_copy(update={"enabled": True}))
    with pytest.raises(RuntimeError, match="raw-page vision"):
        validate_parser_security(unsafe, production_enabled=True)
