"""Synthetic CUDA OCR smoke test. Run only with the intake service idle."""

from __future__ import annotations

import argparse
import gc
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-idle", action="store_true", required=True)
    args = parser.parse_args()
    if not args.confirm_idle:
        parser.error("The shared GPU must be idle")

    import numpy as np
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import RapidOcrOptions
    from docling.models.rapid_ocr_model import RapidOcrModel
    from PIL import Image, ImageDraw, ImageFont
    from rapidocr.inference_engine.pytorch.main import TorchInferSession

    from findociq.ingest.docling_parser import rapidocr_parameters
    from findociq.ingest.gpu_lease import GpuLeaseConfig, VllmGpuLease

    devices: list[str] = []
    original = TorchInferSession.__call__

    def observed(self, image):
        devices.append(str(self.device))
        assert self.device.type == "cuda", "OCR inference unexpectedly used CPU"
        return original(self, image)

    TorchInferSession.__call__ = observed
    lease = VllmGpuLease(GpuLeaseConfig(enabled=True, base_url="http://vllm:8000"))
    with lease.ingestion_batch():
        model = None
        try:
            torch.cuda.reset_peak_memory_stats()
            model = RapidOcrModel(
                enabled=True,
                artifacts_path=None,
                options=RapidOcrOptions(
                    backend="torch", rapidocr_params=rapidocr_parameters("cuda")
                ),
                accelerator_options=AcceleratorOptions(device="cuda"),
            )
            image = Image.new("RGB", (1000, 180), "white")
            ImageDraw.Draw(image).text(
                (30, 50), "SYNTHETIC OCR CHECK 12345", fill="black",
                font=ImageFont.load_default(size=44),
            )
            result = model.reader(np.asarray(image))
            assert devices, "No OCR inference was executed"
            assert "12345" in " ".join(result.txts or ()), "Synthetic OCR failed"
            peak = torch.cuda.max_memory_allocated()
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            TorchInferSession.__call__ = original
    print(json.dumps({"ocr_devices": sorted(set(devices)), "calls": len(devices),
                      "peak_cuda_bytes": peak, "vllm_restored": True}))


if __name__ == "__main__":
    main()
