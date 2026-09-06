"""Authenticated PDF ingestion service behind FinDocIQ's HTTP boundary."""

from __future__ import annotations

import json
import os
from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import asdict
from gc import collect
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Protocol

import fitz

from findociq.api.schema import (
    IngestDocumentRequest,
    IngestDocumentResponse,
)
from findociq.index.embedder import BgeM3Embedder
from findociq.index.store import IndexRecord, QdrantStore
from findociq.ingest.chunker import LayoutAwareChunker
from findociq.ingest.config import IngestionConfig
from findociq.ingest.docling_parser import DocumentParser
from findociq.ingest.gpu_lease import GpuLeaseConfig, VllmGpuLease
from findociq.ingest.router import PageRouter
from findociq.ingest.schema import DocumentBlock, ParsedDocument, TableChunk, TextChunk
from findociq.ingest.vlm_fallback import (
    OpenAICompatibleGemmaVisionExtractor,
    VisionPageExtractor,
)
from findociq.observability.recorder import build_observer
from findociq.observability.schema import ObservabilityConfig
from findociq.retrieve.pipeline import RetrievalRuntimeConfig
from findociq.security.dlp import LocalDlpProvider
from findociq.security.pdf_guard import PdfSafetyScanner, ScanReceipt, UnsafeDocumentError
from findociq.security.policy import ProductionGuardrailPolicy
from findociq.security.storage import EnvelopeEncryptedStore


class DocumentIngestionError(ValueError):
    """Rejected or unparseable document input."""


class IngestionProgressReporter(Protocol):
    def __call__(self, stage: str, message: str, **details: object) -> None: ...


class DocumentIngestionService:
    def __init__(
        self,
        *,
        storage_root: Path,
        parser: DocumentParser,
        chunker: LayoutAwareChunker,
        embedder: BgeM3Embedder,
        store: QdrantStore,
        config_hash: str,
        gpu_lease: VllmGpuLease | None = None,
        max_bytes: int = 25 * 1024 * 1024,
        production_policy: ProductionGuardrailPolicy | None = None,
        scanner: PdfSafetyScanner | None = None,
        dlp: LocalDlpProvider | None = None,
        encrypted_store: EnvelopeEncryptedStore | None = None,
    ) -> None:
        self.storage_root = storage_root.resolve()
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.config_hash = config_hash
        self.gpu_lease = gpu_lease or VllmGpuLease(GpuLeaseConfig(enabled=False))
        self.max_bytes = max_bytes
        self.production_policy = production_policy
        self.scanner = scanner
        self.dlp = dlp
        self.encrypted_store = encrypted_store
        self._lock = Lock()

    def ingest(self, request: IngestDocumentRequest) -> IngestDocumentResponse:
        return self.ingest_batch((request,))[0]

    def ingest_batch(
        self,
        requests: tuple[IngestDocumentRequest, ...],
        *,
        progress: IngestionProgressReporter | None = None,
    ) -> tuple[IngestDocumentResponse, ...]:
        if not requests:
            raise DocumentIngestionError("ingestion batch must contain at least one document")
        count = len(requests)
        if (
            self.production_policy is not None
            and count > self.production_policy.resources.max_batch_documents
        ):
            raise DocumentIngestionError("document batch exceeds configured count limit")
        decoded = tuple(self._decode_and_validate(request) for request in requests)
        if (
            self.production_policy is not None
            and sum(map(len, decoded)) > self.production_policy.resources.max_batch_bytes
        ):
            raise DocumentIngestionError("document batch exceeds configured byte limit")
        self._validate_page_counts(decoded)
        scan_receipts = self._scan_before_gpu(requests, decoded)
        _emit(progress, "waiting_gpu", "Waiting for exclusive GPU access", total=count)
        try:
            with self._lock, self.gpu_lease.ingestion_batch():
                _emit(progress, "gpu_ingestion_ready", "vLLM is sleeping; ingestion owns the GPU")
                try:
                    results = tuple(
                        self._ingest_one(
                            request,
                            document_index=index,
                            document_count=count,
                            progress=progress,
                            content=content,
                            scan_receipt=scan_receipts.get(request.sha256),
                        )
                        for index, (request, content) in enumerate(
                            zip(requests, decoded, strict=True), start=1
                        )
                    )
                finally:
                    _emit(progress, "releasing_gpu", "Releasing ingestion models and GPU memory")
                    self._release_ingestion_models()
            _emit(progress, "vllm_ready", "vLLM is awake and ready for cited extraction")
            return results
        except Exception:
            _emit(progress, "failed", "Document ingestion failed")
            raise

    def _ingest_one(
        self,
        request: IngestDocumentRequest,
        *,
        document_index: int = 1,
        document_count: int = 1,
        progress: IngestionProgressReporter | None = None,
        content: bytes | None = None,
        scan_receipt: ScanReceipt | None = None,
    ) -> IngestDocumentResponse:
        details = {
            "document_name": request.filename,
            "document_index": document_index,
            "document_count": document_count,
        }
        _emit(progress, "validating_document", f"Validating {request.filename}", **details)
        content = content if content is not None else self._decode_and_validate(request)
        digest = sha256(content).hexdigest()

        encrypted_path: Path | None = None
        if self.encrypted_store is not None and request.application_id:
            encrypted_path = self.encrypted_store.store(
                application_id=request.application_id,
                document_id=digest,
                sha256=digest,
                content=content,
            )
            with self.encrypted_store.materialize(encrypted_path, request.filename) as path:
                _emit(
                    progress, "parsing_document", f"Parsing and OCR: {request.filename}", **details
                )
                parsed = self._parse(path)
        else:
            folder = self.storage_root / digest
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / request.filename
            temporary = folder / f".{request.filename}.upload"
            temporary.write_bytes(content)
            temporary.replace(path)
            _emit(
                progress, "parsing_document", f"Parsing and OCR: {request.filename}", **details
            )
            parsed = self._parse(path)
        if (
            self.production_policy is not None
            and len(parsed.pages) > self.production_policy.resources.max_pages
        ):
            raise DocumentIngestionError("document exceeds configured page limit")
        _emit(
            progress,
            "chunking_document",
            f"Creating evidence chunks: {request.filename}",
            completed=len(parsed.pages),
            total=len(parsed.pages),
            **details,
        )
        chunks = tuple(
            chunk.model_copy(
                update={
                    "metadata": {
                        **chunk.metadata,
                        "config_hash": self.config_hash,
                        **(
                            {"application_id": request.application_id}
                            if request.application_id
                            else {}
                        ),
                    }
                }
            )
            for chunk in self.chunker.chunk(parsed)
        )
        if not chunks:
            raise DocumentIngestionError("document produced no evidence chunks")
        dlp_receipts: list[dict[str, object]] = []
        if self.dlp is not None:
            protected = []
            for chunk in chunks:
                redacted, receipts = _redact_chunk(chunk, self.dlp)
                dlp_receipts.extend(receipts)
                protected.append(redacted)
            chunks = tuple(protected)
        _emit(
            progress,
            "embedding_document",
            f"Embedding {len(chunks)} chunks: {request.filename}",
            total=len(chunks),
            **details,
        )
        embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        _emit(
            progress,
            "indexing_document",
            f"Indexing evidence in Qdrant: {request.filename}",
            completed=len(chunks),
            total=len(chunks),
            **details,
        )
        self.store.ensure_collection(self.embedder.dimension)
        self.store.upsert(
            [
                IndexRecord(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
        )
        response = IngestDocumentResponse(
            contract_version=request.contract_version,
            document_id=parsed.document_id,
            filename=request.filename,
            sha256=digest,
            page_count=len(parsed.pages),
            chunk_count=len(chunks),
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            config_hash=self.config_hash,
            application_id=request.application_id,
            policy_hash=request.policy_hash,
            scan_status="clean" if self.scanner is not None else None,
            scan_receipt_sha256=(
                sha256(
                    json.dumps(
                        {
                            **asdict(scan_receipt),
                            "artifact_hash": digest,
                            "policy_hash": request.policy_hash,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if scan_receipt is not None
                else None
            ),
            dlp_receipt_sha256=(
                sha256(
                    json.dumps(dlp_receipts, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if dlp_receipts
                else None
            ),
            storage_receipt_sha256=(
                sha256(
                    "|".join(
                        (
                            request.application_id or "",
                            parsed.document_id,
                            digest,
                            encrypted_path.name if encrypted_path else "local_plaintext",
                            request.policy_hash or "",
                        )
                    ).encode()
                ).hexdigest()
                if request.application_id
                else None
            ),
        )
        _emit(
            progress,
            "document_completed",
            f"Completed {request.filename}",
            completed=document_index,
            total=document_count,
            **details,
        )
        return response

    def _decode_and_validate(self, request: IngestDocumentRequest) -> bytes:
        try:
            content = b64decode(request.content_base64, validate=True)
        except (Base64Error, ValueError) as error:
            raise DocumentIngestionError("document content is not valid base64") from error
        limit = (
            self.production_policy.resources.max_pdf_bytes
            if self.production_policy is not None
            else self.max_bytes
        )
        if len(content) > limit:
            raise DocumentIngestionError("document exceeds the configured size limit")
        if not content.startswith(b"%PDF-"):
            raise DocumentIngestionError("document is not a PDF")
        if sha256(content).hexdigest() != request.sha256:
            raise DocumentIngestionError("document SHA-256 does not match content")
        if self.production_policy is not None:
            if request.contract_version != "2.0":
                raise DocumentIngestionError("production ingestion requires contract version 2.0")
            if request.policy_hash != self.production_policy.policy_hash:
                raise DocumentIngestionError("guardrail policy hash mismatch")
        return content

    def _scan_before_gpu(
        self, requests: tuple[IngestDocumentRequest, ...], contents: tuple[bytes, ...]
    ) -> dict[str, ScanReceipt]:
        if self.production_policy is None:
            return {}
        if self.scanner is None:
            raise DocumentIngestionError("malware_scanner_unavailable")
        receipts: dict[str, ScanReceipt] = {}
        try:
            with TemporaryDirectory(prefix="findociq-quarantine-") as folder:
                root = Path(folder)
                for index, (request, content) in enumerate(
                    zip(requests, contents, strict=True), start=1
                ):
                    candidate = root / f"{index}-{request.sha256}.pdf"
                    candidate.write_bytes(content)
                    try:
                        receipts[request.sha256] = self.scanner.scan(candidate)
                    except UnsafeDocumentError:
                        if self.encrypted_store is not None and request.application_id:
                            self.encrypted_store.store(
                                application_id=request.application_id,
                                document_id=request.sha256,
                                sha256=request.sha256,
                                content=content,
                                state="quarantined",
                            )
                        raise
        except UnsafeDocumentError as error:
            raise DocumentIngestionError(str(error)) from error
        return receipts

    def _validate_page_counts(self, contents: tuple[bytes, ...]) -> None:
        """Reject excessive page counts before any parser, model, or GPU lease is used."""

        if self.production_policy is None:
            return
        maximum = self.production_policy.resources.max_pages
        for content in contents:
            try:
                with fitz.open(stream=content, filetype="pdf") as document:
                    if document.page_count > maximum:
                        raise DocumentIngestionError("document exceeds configured page limit")
            except DocumentIngestionError:
                raise
            except Exception as error:
                raise DocumentIngestionError("malformed_pdf") from error

    def _parse(self, path: Path) -> ParsedDocument:
        if self.production_policy is None:
            return self.parser.parse(path)
        try:
            return self.parser.parse(
                path,
                timeout_seconds=self.production_policy.resources.parse_timeout_seconds,
                max_pages=self.production_policy.resources.max_pages,
            )
        except TimeoutError as error:
            raise DocumentIngestionError("document_parse_timeout") from error

    def _release_ingestion_models(self) -> None:
        self.embedder.release()
        collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _emit(
    reporter: IngestionProgressReporter | None,
    stage: str,
    message: str,
    **details: object,
) -> None:
    if reporter is not None:
        reporter(stage, message, **details)


def _redact_chunk(
    chunk: TextChunk | TableChunk, dlp: LocalDlpProvider
) -> tuple[TextChunk | TableChunk, tuple[dict[str, object], ...]]:
    """Redact every textual field that will be serialized into Qdrant."""

    fields = ["text"]
    if isinstance(chunk, TableChunk):
        fields.extend(("table_text", "caption", "preceding_context"))
    update: dict[str, object] = {}
    receipts: list[dict[str, object]] = []
    for field_name in fields:
        value = getattr(chunk, field_name)
        if value is None:
            continue
        result = dlp.inspect(value)
        if result.receipt.instruction_risk:
            raise DocumentIngestionError("document_prompt_injection_risk")
        update[field_name] = result.redacted_text
        receipts.append(result.receipt.model_dump(mode="json"))
    return chunk.model_copy(update=update), tuple(receipts)


class _SwitchingVisionExtractor(VisionPageExtractor):
    def __init__(self, extractor: VisionPageExtractor, gpu_lease: VllmGpuLease) -> None:
        self.extractor = extractor
        self.gpu_lease = gpu_lease

    def extract_page(
        self, *, pdf_path: str, page_number: int, document_id: str
    ) -> tuple[DocumentBlock, ...]:
        collect()
        try:
            import torch
        except ImportError:
            pass
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with self.gpu_lease.generation_fallback():
            return self.extractor.extract_page(
                pdf_path=pdf_path,
                page_number=page_number,
                document_id=document_id,
            )


def build_ingestion_service(
    *, storage_root: Path, ingestion_config: Path, index_config: Path, retrieval_config: Path
) -> DocumentIngestionService:
    config = IngestionConfig.from_yaml(ingestion_config)
    production_enabled = os.getenv(
        "FINDOCIQ_PRODUCTION_GUARDRAILS_ENABLED", "false"
    ).casefold() in {"1", "true", "yes"}
    policy = (
        ProductionGuardrailPolicy.from_yaml(
            os.getenv("FINDOCIQ_GUARDRAIL_POLICY", "configs/guardrails/production.yaml")
        )
        if production_enabled
        else None
    )
    if policy is not None and (
        config.parser.accelerator_device != "cpu" or config.vision.enabled
    ):
        raise RuntimeError(
            "production ingestion requires CPU parsing and disabled raw-page vision; "
            "enable vision only behind a pre-model image-redaction boundary"
        )
    encrypted_store = None
    if policy is not None:
        key_file = os.getenv("FINDOCIQ_DOCUMENT_MASTER_KEY_FILE")
        if not key_file:
            raise RuntimeError("production encrypted storage requires a master key secret file")
        encrypted_store = EnvelopeEncryptedStore.from_secret_file(
            storage_root,
            key_file,
            os.getenv("FINDOCIQ_DOCUMENT_KEY_VERSION", "v1"),
        )
    gpu_lease = VllmGpuLease(config.gpu_lease)
    observer = build_observer(ObservabilityConfig.from_yaml(config.observability_config))
    vision = OpenAICompatibleGemmaVisionExtractor(config.vision, observer)
    parser = DocumentParser(
        config=config.parser,
        router=PageRouter(config.router),
        vision_extractor=_SwitchingVisionExtractor(vision, gpu_lease),
    )
    runtime = RetrievalRuntimeConfig.from_yaml(index_config, retrieval_config)
    return DocumentIngestionService(
        storage_root=storage_root,
        parser=parser,
        chunker=LayoutAwareChunker(config.chunker),
        embedder=BgeM3Embedder(runtime.embedding),
        store=QdrantStore(runtime.store),
        config_hash=config.config_hash,
        gpu_lease=gpu_lease,
        production_policy=policy,
        scanner=(
            PdfSafetyScanner(
                host=os.getenv("FINDOCIQ_CLAMAV_HOST", "clamav"),
                port=int(os.getenv("FINDOCIQ_CLAMAV_PORT", "3310")),
            )
            if policy is not None
            else None
        ),
        dlp=(LocalDlpProvider(policy.dlp, policy.policy_hash) if policy is not None else None),
        encrypted_store=encrypted_store,
    )
