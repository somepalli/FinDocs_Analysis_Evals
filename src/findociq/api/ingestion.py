"""Authenticated PDF ingestion service behind FinDocIQ's HTTP boundary."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from hashlib import sha256
from pathlib import Path
from threading import Lock

from findociq.api.schema import IngestDocumentRequest, IngestDocumentResponse
from findociq.index.embedder import BgeM3Embedder
from findociq.index.store import IndexRecord, QdrantStore
from findociq.ingest.chunker import LayoutAwareChunker
from findociq.ingest.config import IngestionConfig
from findociq.ingest.docling_parser import DocumentParser
from findociq.ingest.router import PageRouter
from findociq.ingest.vlm_fallback import OpenAICompatibleGemmaVisionExtractor
from findociq.observability.recorder import build_observer
from findociq.observability.schema import ObservabilityConfig
from findociq.retrieve.pipeline import RetrievalRuntimeConfig


class DocumentIngestionError(ValueError):
    """Rejected or unparseable document input."""


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
        max_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self.storage_root = storage_root.resolve()
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.config_hash = config_hash
        self.max_bytes = max_bytes
        self._lock = Lock()

    def ingest(self, request: IngestDocumentRequest) -> IngestDocumentResponse:
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

        with self._lock:
            folder = self.storage_root / digest
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / request.filename
            temporary = folder / f".{request.filename}.upload"
            temporary.write_bytes(content)
            temporary.replace(path)
            parsed = self.parser.parse(path)
            chunks = tuple(
                chunk.model_copy(
                    update={
                        "metadata": {**chunk.metadata, "config_hash": self.config_hash}
                    }
                )
                for chunk in self.chunker.chunk(parsed)
            )
            if not chunks:
                raise DocumentIngestionError("document produced no evidence chunks")
            embeddings = self.embedder.encode([chunk.text for chunk in chunks])
            self.store.ensure_collection(self.embedder.dimension)
            self.store.upsert(
                [
                    IndexRecord(chunk=chunk, embedding=embedding)
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ]
            )
        return IngestDocumentResponse(
            document_id=parsed.document_id,
            filename=request.filename,
            sha256=digest,
            page_count=len(parsed.pages),
            chunk_count=len(chunks),
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            config_hash=self.config_hash,
        )


def build_ingestion_service(
    *, storage_root: Path, ingestion_config: Path, index_config: Path, retrieval_config: Path
) -> DocumentIngestionService:
    config = IngestionConfig.from_yaml(ingestion_config)
    observer = build_observer(ObservabilityConfig.from_yaml(config.observability_config))
    parser = DocumentParser(
        config=config.parser,
        router=PageRouter(config.router),
        vision_extractor=OpenAICompatibleGemmaVisionExtractor(config.vision, observer),
    )
    runtime = RetrievalRuntimeConfig.from_yaml(index_config, retrieval_config)
    return DocumentIngestionService(
        storage_root=storage_root,
        parser=parser,
        chunker=LayoutAwareChunker(config.chunker),
        embedder=BgeM3Embedder(runtime.embedding),
        store=QdrantStore(runtime.store),
        config_hash=config.config_hash,
    )
