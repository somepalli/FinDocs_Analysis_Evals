"""BGE reranking adapter for the final retrieval stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    model_id: str
    revision: str
    snapshot_dir: str | None = None
    use_fp16: bool = True

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("reranker model_id and revision are required")


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> tuple[float, ...]: ...


class BgeReranker:
    """Load a pinned bge-reranker-v2-m3 snapshot lazily."""

    model_name = "BAAI/bge-reranker-v2-m3"

    def __init__(self, config: RerankerConfig, model: Any | None = None) -> None:
        self.config = config
        self._model = model

    def score(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if not documents:
            return ()
        output = self._get_model().compute_score(
            [[query, document] for document in documents], normalize=True
        )
        if isinstance(output, float | int):
            output = [output]
        scores = tuple(float(score) for score in output)
        if len(scores) != len(documents):
            raise ValueError("reranker returned a different number of scores than documents")
        return scores

    def release(self) -> None:
        """Drop model weights before returning GPU ownership to generation."""
        self._model = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as error:
                raise RuntimeError(
                    "reranker dependencies are missing; run `uv sync --extra retrieval`"
                ) from error
            if self.config.snapshot_dir is not None:
                snapshot = Path(self.config.snapshot_dir).resolve()
                if not snapshot.is_dir():
                    raise RuntimeError(f"pinned reranker snapshot is missing: {snapshot}")
            else:
                try:
                    from huggingface_hub import snapshot_download
                except ImportError as error:
                    raise RuntimeError(
                        "Hugging Face Hub is required when no local reranker snapshot is set"
                    ) from error
                snapshot = Path(
                    snapshot_download(
                        repo_id=self.config.model_id,
                        revision=self.config.revision,
                    )
                )
            self._model = FlagReranker(str(snapshot), use_fp16=self.config.use_fp16)
        return self._model
