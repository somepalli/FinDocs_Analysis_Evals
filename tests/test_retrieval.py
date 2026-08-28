from pathlib import Path
from types import SimpleNamespace

import pytest

from findociq.index.embedder import BgeM3Embedder, Embedding, EmbeddingConfig
from findociq.index.store import IndexRecord
from findociq.ingest.schema import BoundingBox, Provenance, TextChunk
from findociq.retrieve.cli import _chunk_files
from findociq.retrieve.hybrid import reciprocal_rank_fusion
from findociq.retrieve.pipeline import (
    RetrievalPipeline,
    RetrievalRuntimeConfig,
    RetrievalStrategyConfig,
)
from findociq.retrieve.rerank import BgeReranker, RerankerConfig
from findociq.retrieve.schema import RetrievalHit

ROOT = Path(__file__).parents[1]


def provenance() -> Provenance:
    return Provenance(
        document_id="doc",
        source_path="filing.pdf",
        page_number=1,
        bbox=BoundingBox(x0=10, y0=10, x1=100, y1=30),
        page_width=595,
        page_height=842,
    )


def chunk(chunk_id: str, text: str) -> TextChunk:
    return TextChunk(chunk_id=chunk_id, text=text, provenance=(provenance(),))


def hit(chunk_id: str, score: float, rank: int, method: str = "dense") -> RetrievalHit:
    return RetrievalHit(
        chunk=chunk(chunk_id, f"text {chunk_id}"),
        score=score,
        rank=rank,
        method=method,
    )


class FakeEmbedder:
    dimension = 2

    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, query: str) -> Embedding:
        self.queries.append(query)
        return Embedding(dense=(1.0, 0.0), sparse={"indices": (1,), "values": (1.0,)})

    def encode(self, texts: list[str]) -> tuple[Embedding, ...]:
        return tuple(
            Embedding(dense=(1.0, 0.0), sparse={"indices": (), "values": ()}) for _ in texts
        )


class FakeStore:
    def __init__(self) -> None:
        self.dense_args: tuple[Embedding, int] | None = None
        self.hybrid_args: tuple[Embedding, int, int, int] | None = None
        self.results = (hit("a", 0.9, 1), hit("b", 0.8, 2), hit("c", 0.7, 3))

    def dense_search(self, query: Embedding, limit: int) -> tuple[RetrievalHit, ...]:
        self.dense_args = (query, limit)
        return self.results[:limit]

    def hybrid_search(
        self, query: Embedding, limit: int, prefetch_limit: int, rrf_k: int
    ) -> tuple[RetrievalHit, ...]:
        self.hybrid_args = (query, limit, prefetch_limit, rrf_k)
        return self.results[:limit]


class FakeReranker:
    model_name = "fake-reranker"

    def score(self, query: str, documents: list[str]) -> tuple[float, ...]:
        assert query == "what changed?"
        assert len(documents) == 3
        return (0.1, 0.9, 0.4)


def strategy(mode: str, rerank_top_k: int | None = None) -> RetrievalStrategyConfig:
    return RetrievalStrategyConfig(
        name="test",
        mode=mode,  # type: ignore[arg-type]
        retrieve_top_k=3,
        rerank_top_k=rerank_top_k,
        prefetch_top_k=5,
        rrf_k=60,
    )


def test_bge_m3_adapter_emits_dense_and_sorted_sparse_vectors() -> None:
    class Model:
        def encode(self, texts: list[str], **_: object) -> dict[str, object]:
            assert texts == ["revenue"]
            return {"dense_vecs": [[0.1, 0.2]], "lexical_weights": [{9: 0.9, 2: 0.2}]}

    embedder = BgeM3Embedder(
        EmbeddingConfig(model_id="BAAI/bge-m3", revision="rev", dense_dimension=2),
        model=Model(),
    )
    embedding = embedder.encode_query("revenue")
    assert embedding.dense == (0.1, 0.2)
    assert embedding.sparse.indices == (2, 9)
    assert embedding.sparse.values == (0.2, 0.9)


def test_bge_m3_uses_existing_pinned_snapshot_without_hub_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    class Model:
        def __init__(self, snapshot: str, **_: object) -> None:
            loaded.append(snapshot)

    monkeypatch.setitem(
        __import__("sys").modules,
        "FlagEmbedding",
        SimpleNamespace(BGEM3FlagModel=Model),
    )
    snapshot = tmp_path / "bge-m3"
    snapshot.mkdir()
    embedder = BgeM3Embedder(
        EmbeddingConfig(
            model_id="BAAI/bge-m3",
            revision="pinned-revision",
            snapshot_dir=str(snapshot),
        )
    )

    embedder._get_model()  # noqa: SLF001

    assert loaded == [str(snapshot.resolve())]


def test_reranker_uses_existing_pinned_snapshot_without_hub_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    class Model:
        def __init__(self, snapshot: str, **_: object) -> None:
            loaded.append(snapshot)

    monkeypatch.setitem(
        __import__("sys").modules,
        "FlagEmbedding",
        SimpleNamespace(FlagReranker=Model),
    )
    snapshot = tmp_path / "reranker"
    snapshot.mkdir()
    reranker = BgeReranker(
        RerankerConfig(
            model_id="BAAI/bge-reranker-v2-m3",
            revision="pinned-revision",
            snapshot_dir=str(snapshot),
        )
    )

    reranker._get_model()  # noqa: SLF001

    assert loaded == [str(snapshot.resolve())]


def test_index_record_preserves_discriminated_chunk_type() -> None:
    record = IndexRecord(
        chunk=chunk("a", "text"),
        embedding=Embedding(dense=(1.0,), sparse={"indices": (), "values": ()}),
    )
    assert record.model_dump(mode="json")["chunk"]["kind"] == "text"


def test_rrf_deduplicates_and_scores_by_rank() -> None:
    fused = reciprocal_rank_fusion(
        [[hit("a", 0.9, 1), hit("b", 0.8, 2)], [hit("b", 0.7, 1), hit("c", 0.6, 2)]],
        k=1,
    )
    assert [item.chunk.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 2 + 1 / 3)
    assert fused[0].method == "hybrid_rrf"


def test_pipeline_runs_dense_strategy_and_keeps_provenance() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()
    results = RetrievalPipeline(strategy("dense"), embedder, store).retrieve("revenue")
    assert store.dense_args is not None and store.dense_args[1] == 3
    assert [item.rank for item in results] == [1, 2, 3]
    assert results[0].chunk.provenance[0].page_number == 1


def test_pipeline_caches_deterministic_query_results() -> None:
    embedder = FakeEmbedder()
    pipeline = RetrievalPipeline(strategy("dense"), embedder, FakeStore())
    assert pipeline.retrieve("revenue") is pipeline.retrieve("revenue")
    assert embedder.queries == ["revenue"]


def test_pipeline_runs_hybrid_strategy_with_rrf_parameters() -> None:
    store = FakeStore()
    RetrievalPipeline(strategy("hybrid_rrf"), FakeEmbedder(), store).retrieve("revenue")
    assert store.hybrid_args is not None
    assert store.hybrid_args[1:] == (3, 5, 60)


def test_pipeline_reranks_top_k_and_keeps_original_retrieval_score() -> None:
    results = RetrievalPipeline(
        strategy("hybrid_rrf", rerank_top_k=2), FakeEmbedder(), FakeStore(), FakeReranker()
    ).retrieve("what changed?")
    assert [item.chunk.chunk_id for item in results] == ["b", "c"]
    assert [item.rank for item in results] == [1, 2]
    assert results[0].method == "fake-reranker"
    assert results[0].retrieval_score == 0.8


def test_all_three_yaml_strategies_load() -> None:
    for name in ("naive", "hybrid", "hybrid_rerank"):
        config = RetrievalStrategyConfig.from_yaml(ROOT / "configs/retrieval" / f"{name}.yaml")
        assert config.name == name
    runtime = RetrievalRuntimeConfig.from_yaml(
        ROOT / "configs/index/default.yaml", ROOT / "configs/retrieval/naive.yaml"
    )
    assert runtime.embedding.model_id == "BAAI/bge-m3"
    assert isinstance(runtime.reranker, RerankerConfig)


def test_chunk_files_expands_directories_in_stable_order(tmp_path: Path) -> None:
    second = tmp_path / "b.jsonl"
    first = tmp_path / "a.jsonl"
    ignored = tmp_path / "notes.txt"
    for path in (second, first, ignored):
        path.write_text("\n", encoding="utf-8")
    assert _chunk_files([tmp_path]) == (first, second)
