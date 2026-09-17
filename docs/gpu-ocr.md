# Local GPU OCR

The Docker ingestion profile requests CUDA for Docling and explicitly enables
RapidOCR's PyTorch CUDA backend. An unavailable CUDA runtime raises
`ocr_cuda_unavailable` instead of silently selecting CPU OCR.

The existing GPU lease sleeps vLLM before the ingestion batch and wakes it after
ingestion releases its models. Run only one API process against this local shared
GPU: the lease lock is process-local, not distributed.

CPU activity is still expected for PDF inspection, rendering, file I/O, and the
PyMuPDF digital-text fast path. Those operations do not imply OCR is on CPU.
GPU utilization can fall between inference calls.

Local CUDA OCR is permitted independently of generative raw-page vision. The
Docker profile keeps vision disabled; production still rejects raw-page vision
without a pre-model image-redaction boundary. Scanning, ownership, and DLP checks
remain required.

Rebuild and recreate the API container to apply device configuration changes.
Do this after active batches finish; an already-created parser retains its old
device setting. Do not run an independent GPU smoke process alongside an active
batch because it cannot acquire the API process's lease lock.
