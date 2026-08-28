import json
from pathlib import Path

import pytest

from evals.run import run_live_reasoning
from evals.schema import AnswerExpectation, CitationTarget, EvalCase
from findociq.ingest.schema import BoundingBox, Provenance, TextChunk
from findociq.observability.recorder import InMemoryRecorder, TraceObserver
from findociq.observability.schema import TraceContext
from findociq.reason.generation import (
    GenerationConfig,
    ModelTierConfig,
    OllamaGemmaClient,
    VllmGemmaClient,
    build_generation_client,
)
from findociq.reason.pass1_extract import Pass1Extractor
from findociq.reason.pipeline import ReasoningPipeline, ReasoningPipelineConfig
from findociq.reason.schema import SourceCitation
from findociq.reason.serve import vllm_command
from findociq.retrieve.schema import RetrievalHit

ROOT = Path(__file__).parents[1]


def citation() -> SourceCitation:
    return SourceCitation(
        document_id="doc-1",
        page_number=2,
        bbox=BoundingBox(x0=10, y0=20, x1=100, y1=40),
    )


def retrieval_hit() -> RetrievalHit:
    provenance = Provenance(
        document_id="doc-1",
        source_path="filing.pdf",
        page_number=2,
        bbox=citation().bbox,
        page_width=595,
        page_height=842,
    )
    return RetrievalHit(
        chunk=TextChunk(
            chunk_id="chunk-1",
            text="Revenue was INR 1,240 crore in FY25.",
            provenance=(provenance,),
        ),
        score=0.9,
        rank=1,
        method="fixture",
    )


class FakeClient:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.stages: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        trace_context: TraceContext | None = None,
        stage: str = "generation",
    ) -> str:
        del trace_context
        self.prompts.append(prompt)
        self.stages.append(stage)
        return self.responses.pop(0)


def extraction_json() -> str:
    return json.dumps(
        {
            "question": "What was revenue?",
            "figures": [
                {
                    "label": "Revenue",
                    "value": "1240",
                    "unit": "INR crore",
                    "period": "FY25",
                    "citation": citation().model_dump(mode="json"),
                }
            ],
            "notes": [],
        }
    )


def answer_json() -> str:
    return json.dumps(
        {
            "answer": "Revenue was INR 1,240 crore in FY25.",
            "citations": [citation().model_dump(mode="json")],
        }
    )


def pass2_answer_json(*evidence_ids: str) -> str:
    return json.dumps(
        {
            "answer": "Revenue was INR 1,240 crore in FY25.",
            "evidence_ids": list(evidence_ids or ("evidence_1",)),
        }
    )


def test_generation_config_is_local_deterministic_and_yaml_backed() -> None:
    config = GenerationConfig.from_yaml(ROOT / "configs/reasoning/gemma_vllm.yaml")
    assert config.backend == "vllm"
    assert config.model_tier == "single_gpu"
    assert config.model_id == "google/gemma-3-12b-it"
    assert config.quantization == "awq-int4"
    assert config.temperature == 0.0
    assert config.seed == 17
    assert config.timeout_seconds == 600
    fallback = GenerationConfig.from_yaml(ROOT / "configs/reasoning/gemma_ollama.yaml")
    assert fallback.backend == "ollama"
    assert fallback.model_id == "gemma3:4b"
    with pytest.raises(ValueError, match="local HTTP"):
        GenerationConfig(
            base_url="https://example.invalid/v1",
            model_id="google/gemma-3-12b-it",
            revision="rev",
        )


def test_generation_config_accepts_internal_vllm_service_endpoint() -> None:
    config = GenerationConfig(
        base_url="http://vllm:8000/v1",
        model_id="google/gemma-3-4b-it",
        revision="8f28faf05c382a2dd81a471090acdb23156eb354",
        backend="vllm",
    )
    assert config.base_url == "http://vllm:8000/v1"


def test_all_generation_tiers_are_pinned_awq_derivatives() -> None:
    tiers = [
        ModelTierConfig.from_yaml(ROOT / "configs/model_tiers" / f"{name}.yaml")
        for name in ("laptop", "single_gpu", "ceiling")
    ]
    assert [tier.name for tier in tiers] == ["laptop", "single_gpu", "ceiling"]
    assert [tier.source_model_id for tier in tiers] == [
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
    ]
    assert all(tier.quantization == "awq-int4" for tier in tiers)
    assert all(len(tier.revision) == 40 for tier in tiers)
    command = vllm_command(tiers[1], executable="python")
    assert command[:3] == ("python", "-m", "vllm.entrypoints.openai.api_server")
    assert command[command.index("--quantization") + 1] == "awq"
    assert command[command.index("--model") + 1] == tiers[1].model_id
    assert command[command.index("--served-model-name") + 1] == tiers[1].source_model_id
    assert command[command.index("--revision") + 1] == tiers[1].revision


def test_generation_factory_selects_concrete_serving_client() -> None:
    vllm = GenerationConfig.from_yaml(ROOT / "configs/reasoning/gemma_vllm.yaml")
    ollama = GenerationConfig.from_yaml(ROOT / "configs/reasoning/gemma_ollama.yaml")
    assert isinstance(build_generation_client(vllm), VllmGemmaClient)
    assert isinstance(build_generation_client(ollama), OllamaGemmaClient)
    with pytest.raises(ValueError, match="requires a vLLM"):
        VllmGemmaClient(ollama)


def test_two_pass_extracts_then_reasons_only_over_structured_json() -> None:
    client = FakeClient(extraction_json(), pass2_answer_json())
    recorder = InMemoryRecorder()
    pipeline = ReasoningPipeline(
        ReasoningPipelineConfig.from_yaml(ROOT / "configs/pipeline/two_pass.yaml"),
        client,
        TraceObserver(recorder),
    )
    result = pipeline.run("What was revenue?", (retrieval_hit(),))
    assert result.mode == "two_pass"
    assert result.extraction is not None
    assert result.extraction.figures[0].value == "1240"
    assert result.answer.citations[0].page_number == 2
    assert "Revenue was INR 1,240" in client.prompts[0]
    assert "Revenue was INR 1,240" not in client.prompts[1]
    assert '"figures"' in client.prompts[1]
    assert '"evidence_id": "evidence_1"' in client.prompts[1]
    assert '"bbox"' not in client.prompts[1]
    assert client.stages == ["generation.pass1", "generation.pass2"]
    assert {event.stage for event in recorder.events} == {
        "reasoning.total",
        "reasoning.pass1",
        "reasoning.pass2",
        "citation_validation",
    }


def test_pass2_retries_unknown_id_and_resolves_canonical_citation() -> None:
    client = FakeClient(
        extraction_json(),
        pass2_answer_json("invented_evidence"),
        pass2_answer_json("evidence_1", "evidence_1"),
    )
    pipeline = ReasoningPipeline(
        ReasoningPipelineConfig.from_yaml(ROOT / "configs/pipeline/two_pass.yaml"),
        client,
    )

    result = pipeline.run("What was revenue?", (retrieval_hit(),))

    assert result.answer.citations == (citation(),)
    assert client.stages == [
        "generation.pass1",
        "generation.pass2",
        "generation.pass2.retry",
    ]
    assert "invented_evidence" in client.prompts[2]


def test_pass2_fails_closed_when_retry_still_selects_unknown_id() -> None:
    client = FakeClient(
        extraction_json(),
        pass2_answer_json("unknown_1"),
        pass2_answer_json("unknown_2"),
    )
    pipeline = ReasoningPipeline(
        ReasoningPipelineConfig.from_yaml(ROOT / "configs/pipeline/two_pass.yaml"),
        client,
    )

    with pytest.raises(ValueError, match="invalid evidence selection after retry"):
        pipeline.run("What was revenue?", (retrieval_hit(),))


def test_single_pass_has_no_intermediate_extraction() -> None:
    client = FakeClient(answer_json())
    pipeline = ReasoningPipeline(
        ReasoningPipelineConfig.from_yaml(ROOT / "configs/pipeline/single_pass.yaml"),
        client,
    )
    result = pipeline.run("What was revenue?", (retrieval_hit(),))
    assert result.mode == "single_pass"
    assert result.extraction is None
    assert result.answer.answer.startswith("Revenue")
    assert "Revenue was INR 1,240" in client.prompts[0]


def test_single_pass_canonicalizes_rounded_bbox_to_exact_provenance() -> None:
    rounded = json.loads(answer_json())
    rounded["citations"][0]["bbox"] = {"x0": 10.01, "y0": 20, "x1": 100, "y1": 40}
    pipeline = ReasoningPipeline(
        ReasoningPipelineConfig.from_yaml(ROOT / "configs/pipeline/single_pass.yaml"),
        FakeClient(json.dumps(rounded)),
    )
    result = pipeline.run("What was revenue?", (retrieval_hit(),))
    assert result.answer.citations[0].bbox == citation().bbox


def test_pass1_accepts_fenced_json_and_rejects_ungrounded_citation() -> None:
    client = FakeClient(f"```json\n{extraction_json()}\n```")
    extraction = Pass1Extractor(client).extract("What was revenue?", (retrieval_hit(),))
    assert extraction.figures[0].citation.document_id == "doc-1"

    bad = json.dumps(
        {
            "question": "q",
            "figures": [
                {
                    "label": "x",
                    "value": "1",
                    "citation": {
                        "document_id": "other",
                        "page_number": 1,
                        "bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                    },
                }
            ],
            "notes": [],
        }
    )
    with pytest.raises(ValueError, match="not present"):
        Pass1Extractor(FakeClient(bad)).extract("q", (retrieval_hit(),))


def test_pass1_repairs_only_uniquely_grounded_schema_omissions() -> None:
    omitted = json.dumps({"figures": [{"label": "Revenue", "value": "1,240", "unit": "INR crore"}]})
    extraction = Pass1Extractor(FakeClient(omitted)).extract(
        "What was revenue?", (retrieval_hit(),)
    )
    assert extraction.question == "What was revenue?"
    assert extraction.figures[0].citation == citation()

    duplicate = retrieval_hit().model_copy(
        update={
            "chunk": retrieval_hit().chunk.model_copy(
                update={
                    "chunk_id": "chunk-2",
                    "provenance": (
                        retrieval_hit().chunk.provenance[0].model_copy(update={"page_number": 3}),
                    ),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="citation"):
        Pass1Extractor(FakeClient(omitted)).extract(
            "What was revenue?", (retrieval_hit(), duplicate)
        )


def test_pass1_repairs_model_bbox_only_when_value_has_unique_evidence() -> None:
    mismatched = json.loads(extraction_json())
    mismatched["figures"][0]["citation"]["bbox"] = {
        "x0": 0,
        "y0": 0,
        "x1": 1,
        "y1": 1,
    }

    extraction = Pass1Extractor(FakeClient(json.dumps(mismatched))).extract(
        "What was revenue?", (retrieval_hit(),)
    )

    assert extraction.figures[0].citation == citation()


def test_pass1_uses_proposed_page_to_disambiguate_repeated_value() -> None:
    duplicate = retrieval_hit().model_copy(
        update={
            "chunk": retrieval_hit().chunk.model_copy(
                update={
                    "chunk_id": "chunk-2",
                    "provenance": (
                        retrieval_hit().chunk.provenance[0].model_copy(
                            update={"page_number": 3}
                        ),
                    ),
                }
            )
        }
    )
    mismatched = json.loads(extraction_json())
    mismatched["figures"][0]["citation"]["bbox"] = {
        "x0": 0,
        "y0": 0,
        "x1": 1,
        "y1": 1,
    }

    extraction = Pass1Extractor(FakeClient(json.dumps(mismatched))).extract(
        "What was revenue?", (retrieval_hit(), duplicate)
    )

    assert extraction.figures[0].citation == citation()


def test_pass1_uses_overlapping_bbox_to_disambiguate_same_page() -> None:
    other_provenance = retrieval_hit().chunk.provenance[0].model_copy(
        update={"bbox": BoundingBox(x0=200, y0=200, x1=300, y1=240)}
    )
    duplicate = retrieval_hit().model_copy(
        update={
            "chunk": retrieval_hit().chunk.model_copy(
                update={"chunk_id": "chunk-2", "provenance": (other_provenance,)}
            )
        }
    )
    approximate = json.loads(extraction_json())
    approximate["figures"][0]["citation"]["bbox"] = {
        "x0": 9,
        "y0": 19,
        "x1": 101,
        "y1": 41,
    }

    extraction = Pass1Extractor(FakeClient(json.dumps(approximate))).extract(
        "What was revenue?", (retrieval_hit(), duplicate)
    )

    assert extraction.figures[0].citation == citation()


def test_pass1_canonicalizes_unique_page_when_value_is_normalized() -> None:
    normalized = json.loads(extraction_json())
    normalized["figures"][0]["value"] = "1.24 thousand"
    normalized["figures"][0]["citation"]["bbox"] = {
        "x0": 0,
        "y0": 0,
        "x1": 1,
        "y1": 1,
    }

    extraction = Pass1Extractor(FakeClient(json.dumps(normalized))).extract(
        "What was revenue?", (retrieval_hit(),)
    )

    assert extraction.figures[0].citation == citation()


def test_pipeline_rejects_empty_evidence_before_model_call() -> None:
    client = FakeClient(answer_json())
    pipeline = ReasoningPipeline(ReasoningPipelineConfig(name="single_pass", passes=1), client)
    with pytest.raises(ValueError, match="retrieved evidence"):
        pipeline.run("q", ())
    assert client.prompts == []


def test_live_reasoning_scores_both_modes_and_records_model_revision() -> None:
    class FakeRetrievalPipeline:
        def retrieve(
            self, _: str, *, trace_context: TraceContext | None = None
        ) -> tuple[RetrievalHit, ...]:
            del trace_context
            return (retrieval_hit(),)

    config = GenerationConfig(
        base_url="http://localhost:8900/v1",
        model_id="google/gemma-3-4b-it",
        revision="pinned-revision",
    )
    results, predictions = run_live_reasoning(
        (
            EvalCase(
                question_id="q1",
                question="What was revenue?",
                relevant_chunk_ids=("chunk-1",),
                expected_answer=AnswerExpectation(answer_type="numeric", value="1240"),
                expected_citations=(
                    CitationTarget.model_validate(citation().model_dump(mode="json")),
                ),
            ),
        ),
        retrieval_pipeline=FakeRetrievalPipeline(),  # type: ignore[arg-type]
        retrieval_strategy="hybrid_rerank",
        generation_client=FakeClient(answer_json(), extraction_json(), pass2_answer_json()),
        generation_config=config,
        generation_config_path=ROOT / "configs/reasoning/gemma_local.yaml",
        reasoning_config_dir=ROOT / "configs/pipeline",
        retrieval_config_path=ROOT / "configs/retrieval/hybrid_rerank.yaml",
        index_config_path=ROOT / "configs/index/default.yaml",
    )
    assert [result.mode for result in results] == ["single_pass", "two_pass"]
    assert all(result.backend == "live" for result in results)
    assert all(result.model_revision == "pinned-revision" for result in results)
    assert results[1].answer.exact_match_accuracy == 1.0
    assert predictions["two_pass"]["q1"].citations[0].bbox == citation().bbox
