"""BGE-M3 dense and sparse embedding adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model_id: str
    revision: str
    snapshot_dir: str | None = None
    dense_dimension: int = 1024
    batch_size: int = 8
    max_length: int = 8192
    use_fp16: bool = True

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("embedding model_id and revision are required")
        for name in ("dense_dimension", "batch_size", "max_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class SparseEmbedding(BaseModel):
    model_config = ConfigDict(frozen=True)

    indices: tuple[int, ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def matching_lengths(self) -> SparseEmbedding:
        if len(self.indices) != len(self.values):
            raise ValueError("sparse embedding indices and values must have equal lengths")
        if tuple(sorted(self.indices)) != self.indices:
            raise ValueError("sparse embedding indices must be sorted")
        return self


class Embedding(BaseModel):
    model_config = ConfigDict(frozen=True)

    dense: tuple[float, ...] = Field(min_length=1)
    sparse: SparseEmbedding


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> tuple[Embedding, ...]: ...

    def encode_query(self, query: str) -> Embedding: ...


class BgeM3Embedder:
    """Load one pinned BGE-M3 snapshot and emit both supported retrieval modes."""

    def __init__(self, config: EmbeddingConfig, model: Any | None = None) -> None:
        self.config = config
        self._model = model

    @property
    def dimension(self) -> int:
        return self.config.dense_dimension

    def encode(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        if not texts:
            return ()
        output = self._get_model().encode(
            list(texts),
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"]
        sparse_vectors = output["lexical_weights"]
        if len(dense_vectors) != len(texts) or len(sparse_vectors) != len(texts):
            raise ValueError("BGE-M3 returned a different number of vectors than inputs")

        embeddings: list[Embedding] = []
        for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True):
            dense_tuple = tuple(float(value) for value in dense)
            if len(dense_tuple) != self.dimension:
                raise ValueError(
                    f"expected dense dimension {self.dimension}, got {len(dense_tuple)}"
                )
            sparse_pairs = sorted(
                (int(token_id), float(weight))
                for token_id, weight in sparse.items()
                if float(weight) != 0.0
            )
            embeddings.append(
                Embedding(
                    dense=dense_tuple,
                    sparse=SparseEmbedding(
                        indices=tuple(item[0] for item in sparse_pairs),
                        values=tuple(item[1] for item in sparse_pairs),
                    ),
                )
            )
        return tuple(embeddings)

    def encode_query(self, query: str) -> Embedding:
        if not query.strip():
            raise ValueError("query must not be blank")
        return self.encode((query,))[0]

    def release(self) -> None:
        """Release the lazy model before GPU ownership returns to vLLM."""

        self._model = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
                from huggingface_hub import snapshot_download
            except ImportError as error:
                raise RuntimeError(
                    "BGE-M3 dependencies are missing; run `uv sync --extra retrieval`"
                ) from error
            snapshot = Path(
                snapshot_download(
                    repo_id=self.config.model_id,
                    revision=self.config.revision,
                    local_dir=self.config.snapshot_dir,
                )
            )
            self._model = BGEM3FlagModel(
                str(snapshot),
                use_fp16=self.config.use_fp16,
            )
        return self._model
