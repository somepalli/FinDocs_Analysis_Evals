"""Authenticated PDF ingestion service behind FinDocIQ's HTTP boundary."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from gc import collect
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Protocol

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
from findociq.ingest.schema import DocumentBlock
from findociq.ingest.vlm_fallback import (
    OpenAICompatibleGemmaVisionExtractor,
    VisionPageExtractor,
)
from findociq.observability.recorder import build_observer
from findociq.observability.schema import ObservabilityConfig
from findociq.retrieve.pipeline import RetrievalRuntimeConfig


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
    ) -> None:
        self.storage_root = storage_root.resolve()
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.config_hash = config_hash
        self.gpu_lease = gpu_lease or VllmGpuLease(GpuLeaseConfig(enabled=False))
        self.max_bytes = max_bytes
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
                        )
                        for index, request in enumerate(requests, start=1)
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
    ) -> IngestDocumentResponse:
        details = {
            "document_name": request.filename,
            "document_index": document_index,
            "document_count": document_count,
        }
        _emit(progress, "validating_document", f"Validating {request.filename}", **details)
        try:
            content = b64decode(request.content_base64, validate=True)
        except (Base64Error, ValueError) as error:
            raise DocumentIngestionError("document content is not valid base64") from error
        if len(content) > self.max_bytes:
            raise DocumentIngestionError("document exceeds the configured size limit")
        if not content.startswith(b"%PDF-"):
            raise DocumentIngestionError("document is not a PDF")
        digest = sha256(content).hexdigest()
        if digest != request.sha256:
            raise DocumentIngestionError("document SHA-256 does not match content")

        folder = self.storage_root / digest
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / request.filename
        temporary = folder / f".{request.filename}.upload"
        temporary.write_bytes(content)
        temporary.replace(path)
        _emit(progress, "parsing_document", f"Parsing and OCR: {request.filename}", **details)
        parsed = self.parser.parse(path)
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
                update={"metadata": {**chunk.metadata, "config_hash": self.config_hash}}
            )
            for chunk in self.chunker.chunk(parsed)
        )
        if not chunks:
            raise DocumentIngestionError("document produced no evidence chunks")
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
            document_id=parsed.document_id,
            filename=request.filename,
            sha256=digest,
            page_count=len(parsed.pages),
            chunk_count=len(chunks),
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            config_hash=self.config_hash,
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
    )
