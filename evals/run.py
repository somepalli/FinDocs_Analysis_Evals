"""Run reproducible retrieval sweeps and write separate score tables.

By default, ``--sweep`` uses the checked-in fixture records so the command is
immediately runnable without downloading model weights. Pass ``--live`` to
run each strategy against local Qdrant and the pinned BGE models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from evals.cross_lingual import (
    aggregate_answers_by_language,
    aggregate_retrieval_by_language,
    validate_paired_cases,
)
from evals.schema import (
    AnswerPrediction,
    CitationTarget,
    EvalCase,
    ReasoningPrediction,
    ReasoningResult,
    RetrievedItem,
    StrategyResult,
    SweepResult,
)
from evals.scorers.answers import aggregate_answers
from evals.scorers.retrieval import aggregate_retrieval
from findociq.index.embedder import BgeM3Embedder
from findociq.index.store import QdrantStore
from findociq.observability.aggregate import aggregate_traces, write_observability_report
from findociq.observability.recorder import (
    TraceObserver,
    build_observer,
    load_trace_events,
)
from findociq.observability.schema import ObservabilityConfig, TraceContext
from findociq.reason.generation import (
    GenerationClient,
    GenerationConfig,
    build_generation_client,
)
from findociq.reason.pipeline import ReasoningPipeline, ReasoningPipelineConfig
from findociq.retrieve.pipeline import (
    RetrievalPipeline,
    RetrievalRuntimeConfig,
)
from findociq.retrieve.rerank import BgeReranker

CASE_ADAPTER = TypeAdapter(EvalCase)
ANSWER_ADAPTER = TypeAdapter(AnswerPrediction)
STRATEGIES = ("naive", "hybrid", "hybrid_rerank")


def load_cases(path: str | Path) -> tuple[EvalCase, ...]:
    source = Path(path)
    cases: list[EvalCase] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    cases.append(CASE_ADAPTER.validate_json(line))
                except ValueError as error:
                    raise ValueError(f"invalid eval case at {source}:{line_number}") from error
    if not cases:
        raise ValueError(f"evaluation dataset is empty: {source}")
    identifiers = [case.question_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evaluation question_id values must be unique")
    return tuple(cases)


def load_retrieval_records(path: str | Path) -> dict[str, dict[str, tuple[RetrievedItem, ...]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("retrieval records must be an object keyed by strategy")
    output: dict[str, dict[str, tuple[RetrievedItem, ...]]] = {}
    for strategy, questions in payload.items():
        if not isinstance(questions, dict):
            raise ValueError(f"retrieval records for {strategy} must be an object")
        output[strategy] = {
            question_id: tuple(RetrievedItem.model_validate(item) for item in items)
            for question_id, items in questions.items()
        }
    return output


def load_answer_records(path: str | Path) -> dict[str, dict[str, AnswerPrediction]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("answer records must be an object keyed by strategy")
    return {
        strategy: {
            question_id: ANSWER_ADAPTER.validate_python(prediction)
            for question_id, prediction in questions.items()
        }
        for strategy, questions in payload.items()
    }


def run_sweep(
    cases: Sequence[EvalCase],
    *,
    strategy_paths: dict[str, Path],
    dataset_sha256: str,
    retrieval_records: dict[str, dict[str, tuple[RetrievedItem, ...]]] | None = None,
    answer_records: dict[str, dict[str, AnswerPrediction]] | None = None,
    reasoning_records: dict[str, dict[str, AnswerPrediction]] | None = None,
    pipeline_factory: Callable[[str, Path], RetrievalPipeline] | None = None,
    index_config_path: Path = Path("configs/index/default.yaml"),
    reasoning_config_dir: Path = Path("configs/pipeline"),
) -> SweepResult:
    results: list[StrategyResult] = []
    for strategy, config_path in strategy_paths.items():
        if retrieval_records is not None:
            predictions = retrieval_records.get(strategy, {})
            backend = "fixture"
        else:
            if pipeline_factory is None:
                pipeline_factory = _default_pipeline_factory(index_config_path)
            pipeline = pipeline_factory(strategy, config_path)
            strategy_hash = file_hash(index_config_path, config_path)
            predictions = _retrieve_live(
                cases,
                pipeline,
                strategy=strategy,
                config_hash=strategy_hash,
                dataset_sha256=dataset_sha256,
            )
            backend = "live"
        answer = None
        if answer_records is not None and strategy in answer_records:
            answer = aggregate_answers(cases, answer_records[strategy])
        results.append(
            StrategyResult(
                strategy=strategy,
                config_hash=file_hash(index_config_path, config_path),
                backend=backend,
                retrieval=aggregate_retrieval(cases, predictions),
                answer=answer,
                retrieval_by_language=aggregate_retrieval_by_language(cases, predictions),
                answer_by_language=(
                    aggregate_answers_by_language(cases, answer_records[strategy])
                    if answer_records is not None and strategy in answer_records
                    else ()
                ),
            )
        )
    reasoning: list[ReasoningResult] = []
    if reasoning_records is not None:
        for mode, predictions in reasoning_records.items():
            reasoning.append(
                ReasoningResult(
                    mode=mode,
                    config_hash=file_hash(
                        reasoning_config_dir / f"{mode}.yaml",
                        *_reasoning_prompt_paths(mode),
                    ),
                    backend="fixture",
                    answer=aggregate_answers(cases, predictions),
                    answer_by_language=aggregate_answers_by_language(cases, predictions),
                )
            )
    return SweepResult(
        dataset_sha256=dataset_sha256,
        results=tuple(results),
        reasoning=tuple(reasoning),
    )


def run_live_reasoning(
    cases: Sequence[EvalCase],
    *,
    retrieval_pipeline: RetrievalPipeline,
    retrieval_strategy: str,
    generation_client: GenerationClient,
    generation_config: GenerationConfig,
    generation_config_path: Path,
    reasoning_config_dir: Path,
    retrieval_config_path: Path,
    index_config_path: Path,
    observer: TraceObserver | None = None,
    dataset_sha256: str | None = None,
) -> tuple[tuple[ReasoningResult, ...], dict[str, dict[str, ReasoningPrediction]]]:
    active_observer = observer or TraceObserver()
    retrieval_hash = file_hash(index_config_path, retrieval_config_path)
    hits = {
        case.question_id: retrieval_pipeline.retrieve(
            case.question,
            trace_context=TraceContext.for_query(
                case.question,
                operation=f"reasoning_evidence:{retrieval_strategy}",
                question_id=case.question_id,
                config_hash=retrieval_hash,
                dataset_sha256=dataset_sha256,
            ),
        )
        for case in cases
    }
    results: list[ReasoningResult] = []
    all_predictions: dict[str, dict[str, ReasoningPrediction]] = {}
    for mode in ("single_pass", "two_pass"):
        pipeline_path = reasoning_config_dir / f"{mode}.yaml"
        pipeline = ReasoningPipeline(
            ReasoningPipelineConfig.from_yaml(pipeline_path),
            generation_client,
            active_observer,
        )
        reasoning_hash = file_hash(
            pipeline_path,
            *_generation_config_paths(generation_config_path),
            retrieval_config_path,
            index_config_path,
            *_reasoning_prompt_paths(mode),
        )
        predictions: dict[str, ReasoningPrediction] = {}
        for case in cases:
            trace_context = TraceContext.for_query(
                case.question,
                operation=f"reasoning:{mode}",
                question_id=case.question_id,
                config_hash=reasoning_hash,
                dataset_sha256=dataset_sha256,
            )
            try:
                run = pipeline.run(
                    case.question,
                    hits[case.question_id],
                    trace_context=trace_context,
                )
                prediction = ReasoningPrediction(
                    answer=run.answer.answer,
                    citations=tuple(
                        CitationTarget.model_validate(item.model_dump(mode="json"))
                        for item in run.answer.citations
                    ),
                )
            except (RuntimeError, ValueError) as error:
                prediction = ReasoningPrediction(
                    answer="",
                    citations=(),
                    error=f"{type(error).__name__}: {error}",
                )
            predictions[case.question_id] = prediction
        all_predictions[mode] = predictions
        results.append(
            ReasoningResult(
                mode=mode,
                config_hash=reasoning_hash,
                backend="live",
                model_id=generation_config.model_id,
                model_revision=generation_config.revision,
                retrieval_strategy=retrieval_strategy,
                answer=aggregate_answers(cases, predictions),
                answer_by_language=aggregate_answers_by_language(cases, predictions),
            )
        )
    return tuple(results), all_predictions


def write_reasoning_predictions(
    records: dict[str, dict[str, ReasoningPrediction]], output_dir: str | Path
) -> Path:
    path = Path(output_dir) / "reasoning_predictions.json"
    payload = {
        mode: {
            question_id: prediction.model_dump(mode="json")
            for question_id, prediction in predictions.items()
        }
        for mode, predictions in records.items()
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_results(result: SweepResult, output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "retrieval_sweep.json"
    markdown_path = destination / "retrieval_sweep.md"
    json_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(result: SweepResult) -> str:
    lines = [
        "# FinDocIQ retrieval sweep",
        "",
        f"Dataset SHA-256: `{result.dataset_sha256}`",
        "",
        "## Retrieval quality",
        "",
        "| Strategy | Backend | Queries | Recall@1 | Recall@5 | Recall@8 | MRR | nDCG@8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result.results:
        metrics = item.retrieval
        lines.append(
            f"| {item.strategy} | {item.backend} | {metrics.query_count} | "
            f"{metrics.recall_at_1:.3f} | {metrics.recall_at_5:.3f} | "
            f"{metrics.recall_at_8:.3f} | {metrics.mrr:.3f} | {metrics.ndcg_at_8:.3f} |"
        )
    answers = [item for item in result.results if item.answer is not None]
    if answers:
        lines.extend(
            [
                "",
                "## Answer quality",
                "",
                "| Strategy | Answers | Exact | Numeric exact | Text exact | "
                "Citation P | Citation R | Citation F1 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in answers:
            assert item.answer is not None
            metrics = item.answer
            lines.append(
                f"| {item.strategy} | {metrics.answer_count} | "
                f"{metrics.exact_match_accuracy:.3f} | "
                f"{_optional_metric(metrics.numeric_exact_accuracy)} | "
                f"{_optional_metric(metrics.text_exact_accuracy)} | "
                f"{metrics.citation_precision:.3f} | {metrics.citation_recall:.3f} | "
                f"{metrics.citation_f1:.3f} |"
            )
    if result.reasoning:
        lines.extend(
            [
                "",
                "## Reasoning comparison",
                "",
                "| Mode | Backend | Model | Retrieval | Answers | Exact | "
                "Numeric exact | Citation F1 |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in result.reasoning:
            metrics = item.answer
            lines.append(
                f"| {item.mode} | {item.backend} | {item.model_id or '-'} | "
                f"{item.retrieval_strategy or '-'} | {metrics.answer_count} | "
                f"{metrics.exact_match_accuracy:.3f} | "
                f"{_optional_metric(metrics.numeric_exact_accuracy)} | "
                f"{metrics.citation_f1:.3f} |"
            )
    if any(len(item.retrieval_by_language) > 1 for item in result.results):
        lines.extend(
            [
                "",
                "## Retrieval quality by query language",
                "",
                "| Strategy | Language | Queries | Recall@1 | Recall@5 | Recall@8 | MRR | nDCG@8 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in result.results:
            for language_result in item.retrieval_by_language:
                metrics = language_result.retrieval
                lines.append(
                    f"| {item.strategy} | {language_result.language} | {metrics.query_count} | "
                    f"{metrics.recall_at_1:.3f} | {metrics.recall_at_5:.3f} | "
                    f"{metrics.recall_at_8:.3f} | {metrics.mrr:.3f} | "
                    f"{metrics.ndcg_at_8:.3f} |"
                )
        lines.extend(
            [
                "",
                "### Hindi minus English retrieval gap",
                "",
                "| Strategy | Recall@1 gap | Recall@5 gap | MRR gap | nDCG@8 gap |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in result.results:
            languages = {entry.language: entry.retrieval for entry in item.retrieval_by_language}
            if "en" in languages and "hi" in languages:
                english, hindi = languages["en"], languages["hi"]
                lines.append(
                    f"| {item.strategy} | {hindi.recall_at_1 - english.recall_at_1:+.3f} | "
                    f"{hindi.recall_at_5 - english.recall_at_5:+.3f} | "
                    f"{hindi.mrr - english.mrr:+.3f} | "
                    f"{hindi.ndcg_at_8 - english.ndcg_at_8:+.3f} |"
                )
    if any(len(item.answer_by_language) > 1 for item in result.reasoning):
        lines.extend(
            [
                "",
                "## Reasoning quality by query language",
                "",
                "| Mode | Language | Answers | Exact | Numeric exact | Citation F1 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in result.reasoning:
            for language_result in item.answer_by_language:
                metrics = language_result.answer
                lines.append(
                    f"| {item.mode} | {language_result.language} | {metrics.answer_count} | "
                    f"{metrics.exact_match_accuracy:.3f} | "
                    f"{_optional_metric(metrics.numeric_exact_accuracy)} | "
                    f"{metrics.citation_f1:.3f} |"
                )
        lines.extend(
            [
                "",
                "### Hindi minus English reasoning gap",
                "",
                "| Mode | Numeric exact gap | Citation F1 gap |",
                "|---|---:|---:|",
            ]
        )
        for item in result.reasoning:
            languages = {entry.language: entry.answer for entry in item.answer_by_language}
            if "en" in languages and "hi" in languages:
                english, hindi = languages["en"], languages["hi"]
                numeric_gap = (
                    "-"
                    if english.numeric_exact_accuracy is None
                    or hindi.numeric_exact_accuracy is None
                    else f"{hindi.numeric_exact_accuracy - english.numeric_exact_accuracy:+.3f}"
                )
                lines.append(
                    f"| {item.mode} | {numeric_gap} | "
                    f"{hindi.citation_f1 - english.citation_f1:+.3f} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.sweep:
        raise SystemExit("pass --sweep to run the evaluation sweep")
    cases = load_cases(args.dataset)
    if args.validate_cross_lingual_pairs:
        validate_paired_cases(cases)
    records = None if args.live else load_retrieval_records(args.retrieval_records)
    answers = load_answer_records(args.answer_predictions) if args.answer_predictions else None
    reasoning = (
        load_answer_records(args.reasoning_predictions) if args.reasoning_predictions else None
    )
    if args.live_reasoning and not args.live:
        raise SystemExit("--live-reasoning requires --live")
    if args.live_reasoning and reasoning is not None:
        raise SystemExit("use either --live-reasoning or --reasoning-predictions")
    observability_config = (
        ObservabilityConfig.from_yaml(args.observability_config)
        if args.observability_config
        else ObservabilityConfig()
    )
    observer = build_observer(observability_config, reset=True)
    strategy_paths = {name: args.config_dir / f"{name}.yaml" for name in STRATEGIES}
    pipeline_factory = _default_pipeline_factory(args.index_config, observer) if args.live else None
    result = run_sweep(
        cases,
        strategy_paths=strategy_paths,
        dataset_sha256=file_hash(args.dataset),
        retrieval_records=records,
        answer_records=answers,
        reasoning_records=reasoning,
        pipeline_factory=pipeline_factory,
        index_config_path=args.index_config,
        reasoning_config_dir=args.reasoning_config_dir,
    )
    live_predictions = None
    if args.live_reasoning:
        generation_config = GenerationConfig.from_yaml(args.generation_config)
        retrieval_config = args.config_dir / f"{args.reasoning_retrieval_strategy}.yaml"
        assert pipeline_factory is not None
        live_reasoning, live_predictions = run_live_reasoning(
            cases,
            retrieval_pipeline=pipeline_factory(
                args.reasoning_retrieval_strategy,
                retrieval_config,
            ),
            retrieval_strategy=args.reasoning_retrieval_strategy,
            generation_client=build_generation_client(generation_config, observer),
            generation_config=generation_config,
            generation_config_path=args.generation_config,
            reasoning_config_dir=args.reasoning_config_dir,
            retrieval_config_path=retrieval_config,
            index_config_path=args.index_config,
            observer=observer,
            dataset_sha256=file_hash(args.dataset),
        )
        result = result.model_copy(update={"reasoning": live_reasoning})
    json_path, markdown_path = write_results(result, args.results_dir)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    if live_predictions is not None:
        print(f"wrote {write_reasoning_predictions(live_predictions, args.results_dir)}")
    if observability_config.enabled:
        summary = aggregate_traces(load_trace_events(observability_config.trace_path))
        summary_paths = write_observability_report(summary, args.results_dir)
        print(f"wrote {summary_paths[0]}")
        print(f"wrote {summary_paths[1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--live", action="store_true", help="use local Qdrant and BGE models")
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/smoke.jsonl"))
    parser.add_argument(
        "--retrieval-records",
        type=Path,
        default=Path("evals/datasets/smoke_retrieval.json"),
    )
    parser.add_argument("--answer-predictions", type=Path)
    parser.add_argument("--reasoning-predictions", type=Path)
    parser.add_argument("--live-reasoning", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=Path("evals/results"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/retrieval"))
    parser.add_argument("--index-config", type=Path, default=Path("configs/index/default.yaml"))
    parser.add_argument("--reasoning-config-dir", type=Path, default=Path("configs/pipeline"))
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/reasoning/gemma_vllm.yaml"),
    )
    parser.add_argument(
        "--reasoning-retrieval-strategy",
        choices=STRATEGIES,
        default="hybrid_rerank",
    )
    parser.add_argument(
        "--observability-config",
        type=Path,
        help="enable typed local JSONL tracing with this YAML configuration",
    )
    parser.add_argument(
        "--validate-cross-lingual-pairs",
        action="store_true",
        help="require matched English-Hindi cases with identical scoring targets",
    )
    return parser.parse_args()


def file_hash(*paths: str | Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _retrieve_live(
    cases: Iterable[EvalCase],
    pipeline: RetrievalPipeline,
    *,
    strategy: str,
    config_hash: str,
    dataset_sha256: str,
) -> dict[str, tuple[RetrievedItem, ...]]:
    return {
        case.question_id: tuple(
            RetrievedItem(chunk_id=hit.chunk.chunk_id, rank=hit.rank, score=hit.score)
            for hit in pipeline.retrieve(
                case.question,
                trace_context=TraceContext.for_query(
                    case.question,
                    operation=f"retrieval:{strategy}",
                    question_id=case.question_id,
                    config_hash=config_hash,
                    dataset_sha256=dataset_sha256,
                ),
            )
        )
        for case in cases
    }


def _default_pipeline_factory(
    index_path: Path, observer: TraceObserver | None = None
) -> Callable[[str, Path], RetrievalPipeline]:
    embedder: BgeM3Embedder | None = None
    store: QdrantStore | None = None
    reranker: BgeReranker | None = None
    pipelines: dict[str, RetrievalPipeline] = {}

    def factory(strategy: str, strategy_path: Path) -> RetrievalPipeline:
        nonlocal embedder, store, reranker
        if strategy in pipelines:
            return pipelines[strategy]
        runtime = RetrievalRuntimeConfig.from_yaml(index_path, strategy_path)
        if embedder is None:
            embedder = BgeM3Embedder(runtime.embedding)
        if store is None:
            store = QdrantStore(runtime.store)
        selected_reranker = None
        if runtime.strategy.rerank_top_k is not None:
            if reranker is None:
                reranker = BgeReranker(runtime.reranker)
            selected_reranker = reranker
        pipelines[strategy] = RetrievalPipeline(
            runtime.strategy,
            embedder,
            store,
            selected_reranker,
            observer,
        )
        return pipelines[strategy]

    return factory


def _optional_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _reasoning_prompt_paths(mode: str) -> tuple[Path, ...]:
    prompt_dir = Path("src/findociq/reason/prompts")
    if mode == "single_pass":
        return (prompt_dir / "single_pass_reason.txt",)
    if mode == "two_pass":
        return (
            prompt_dir / "pass1_extract.txt",
            prompt_dir / "pass2_reason.txt",
            prompt_dir / "pass2_retry.txt",
        )
    raise ValueError(f"unsupported reasoning mode: {mode}")


def _generation_config_paths(path: str | Path) -> tuple[Path, ...]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"generation config must be a mapping: {source}")
    tier_path = payload.get("tier_config")
    if tier_path is None:
        return (source,)
    return (source, (source.parent / str(tier_path)).resolve())


if __name__ == "__main__":
    main()
