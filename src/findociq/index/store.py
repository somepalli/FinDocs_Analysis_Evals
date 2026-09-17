"""Qdrant named-vector storage for BGE-M3 dense and sparse embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from findociq.index.embedder import Embedding
from findociq.ingest.schema import Chunk, TableChunk, TextChunk
from findociq.retrieve.schema import RetrievalHit

CHUNK_ADAPTER = TypeAdapter(Chunk)


@dataclass(frozen=True, slots=True)
class QdrantStoreConfig:
    url: str = "http://localhost:6999"
    collection: str = "findociq_chunks"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.url or not self.collection:
            raise ValueError("Qdrant url and collection are required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class IndexRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: TextChunk | TableChunk = Field(discriminator="kind")
    embedding: Embedding


class RetrievalStore(Protocol):
    def ensure_collection(self, dense_dimension: int) -> None: ...

    def upsert(self, records: Sequence[IndexRecord]) -> None: ...

    def dense_search(
        self,
        query: Embedding,
        limit: int,
        document_ids: tuple[str, ...] = (),
        application_id: str | None = None,
    ) -> tuple[RetrievalHit, ...]: ...

    def hybrid_search(
        self,
        query: Embedding,
        limit: int,
        prefetch_limit: int,
        rrf_k: int,
        document_ids: tuple[str, ...] = (),
        application_id: str | None = None,
    ) -> tuple[RetrievalHit, ...]: ...


class QdrantStore:
    def __init__(self, config: QdrantStoreConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client

    def ensure_collection(self, dense_dimension: int) -> None:
        client, models = self._dependencies()
        if client.collection_exists(self.config.collection):
            return
        client.create_collection(
            collection_name=self.config.collection,
            vectors_config={
                self.config.dense_vector_name: models.VectorParams(
                    size=dense_dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    def upsert(self, records: Sequence[IndexRecord]) -> None:
        if not records:
            return
        client, models = self._dependencies()
        points = [
            models.PointStruct(
                id=self.point_id(record.chunk),
                vector={
                    self.config.dense_vector_name: list(record.embedding.dense),
                    self.config.sparse_vector_name: models.SparseVector(
                        indices=list(record.embedding.sparse.indices),
                        values=list(record.embedding.sparse.values),
                    ),
                },
                payload={"chunk": record.chunk.model_dump(mode="json")},
            )
            for record in records
        ]
        client.upsert(collection_name=self.config.collection, points=points, wait=True)

    @staticmethod
    def point_id(chunk: TextChunk | TableChunk) -> str:
        """Keep identical document chunks isolated between applications."""

        application_id = chunk.metadata.get("application_id")
        identity = (
            f"{application_id}:{chunk.chunk_id}"
            if isinstance(application_id, str) and application_id
            else chunk.chunk_id
        )
        return str(uuid5(NAMESPACE_URL, identity))

    def scoped_chunks(
        self,
        document_ids: tuple[str, ...],
        application_id: str | None = None,
        *,
        max_chunks: int = 20000,
    ) -> tuple[Chunk, ...]:
        """Read bounded source evidence; never silently use a truncated inventory."""
        if not document_ids:
            raise ValueError("evidence_document_scope_missing")
        client, models = self._dependencies()
        chunks = []
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=self.config.collection,
                scroll_filter=self._document_filter(models, document_ids, application_id),
                limit=min(256, max_chunks + 1 - len(chunks)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                chunk = CHUNK_ADAPTER.validate_python((record.payload or {}).get("chunk"))
                if any(p.document_id not in document_ids for p in chunk.provenance):
                    raise PermissionError("evidence_scope_violation")
                if application_id and chunk.metadata.get("application_id") != application_id:
                    raise PermissionError("evidence_scope_violation")
                chunks.append(chunk)
            if len(chunks) > max_chunks:
                raise ValueError("evidence_inventory_limit")
            if offset is None:
                break
        present = {p.document_id for c in chunks for p in c.provenance}
        if present != set(document_ids):
            raise ValueError("evidence_document_missing")
        return tuple(chunks)

    def delete_application(self, application_id: str) -> None:
        client, models = self._dependencies()
        client.delete(
            collection_name=self.config.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk.metadata.application_id",
                            match=models.MatchValue(value=application_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def dense_search(
        self,
        query: Embedding,
        limit: int,
        document_ids: tuple[str, ...] = (),
        application_id: str | None = None,
    ) -> tuple[RetrievalHit, ...]:
        client, models = self._dependencies()
        response = client.query_points(
            collection_name=self.config.collection,
            query=list(query.dense),
            using=self.config.dense_vector_name,
            limit=limit,
            with_payload=True,
            query_filter=self._document_filter(models, document_ids, application_id),
        )
        return self._hits(response.points, "dense")

    def hybrid_search(
        self,
        query: Embedding,
        limit: int,
        prefetch_limit: int,
        rrf_k: int,
        document_ids: tuple[str, ...] = (),
        application_id: str | None = None,
    ) -> tuple[RetrievalHit, ...]:
        client, models = self._dependencies()
        response = client.query_points(
            collection_name=self.config.collection,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(query.sparse.indices),
                        values=list(query.sparse.values),
                    ),
                    using=self.config.sparse_vector_name,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=list(query.dense),
                    using=self.config.dense_vector_name,
                    limit=prefetch_limit,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
            limit=limit,
            with_payload=True,
            query_filter=self._document_filter(models, document_ids, application_id),
        )
        return self._hits(response.points, "hybrid_rrf")

    @staticmethod
    def _document_filter(
        models: Any, document_ids: tuple[str, ...], application_id: str | None = None
    ) -> Any | None:
        if not document_ids and not application_id:
            return None
        conditions = []
        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key="chunk.provenance[].document_id",
                    match=models.MatchAny(any=list(document_ids)),
                )
            )
        if application_id:
            conditions.append(
                models.FieldCondition(
                    key="chunk.metadata.application_id",
                    match=models.MatchValue(value=application_id),
                )
            )
        return models.Filter(must=conditions)

    def _dependencies(self) -> tuple[Any, Any]:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as error:
            raise RuntimeError(
                "Qdrant dependencies are missing; run `uv sync --extra retrieval`"
            ) from error
        if self._client is None:
            self._client = QdrantClient(
                url=self.config.url,
                timeout=self.config.timeout_seconds,
            )
        return self._client, models

    @staticmethod
    def _hits(points: Sequence[Any], method: str) -> tuple[RetrievalHit, ...]:
        hits: list[RetrievalHit] = []
        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            chunk = CHUNK_ADAPTER.validate_python(payload.get("chunk"))
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=float(point.score),
                    rank=rank,
                    method=method,
                )
            )
        return tuple(hits)
