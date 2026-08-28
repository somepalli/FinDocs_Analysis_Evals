from pathlib import Path

from fastapi.testclient import TestClient

from findociq.api.app import create_app
from findociq.api.schema import IngestDocumentResponse
from findociq.ingest.schema import BoundingBox
from findociq.reason.schema import (
    ExtractedFigure,
    Pass1Extraction,
    ReasonedAnswer,
    ReasoningRun,
    SourceCitation,
)
from findociq.service import ApiConfig

ROOT = Path(__file__).parents[1]


class FakeQueryService:
    default_mode = "two_pass"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, str | None]] = []

    def query(self, question: str, *, mode: str, question_id: str | None = None) -> ReasoningRun:
        self.calls.append((question, mode, question_id))
        if self.fail:
            raise RuntimeError("secret backend detail")
        source = SourceCitation(
            document_id="doc-1",
            page_number=2,
            bbox=BoundingBox(x0=1, y0=2, x1=3, y1=4),
        )
        extraction = (
            Pass1Extraction(
                question=question,
                figures=(
                    ExtractedFigure(
                        label="Revenue",
                        value="10",
                        unit="INR crore",
                        period="FY2025",
                        citation=source,
                    ),
                ),
                notes=("Consolidated revenue",),
            )
            if mode == "two_pass"
            else None
        )
        return ReasoningRun(
            mode=mode,
            question=question,
            answer=ReasonedAnswer(
                answer="Revenue was Rs. 10 crore.",
                citations=(source,),
            ),
            extraction=extraction,
        )


class FakeIngestionService:
    def __init__(self) -> None:
        self.calls = []

    def ingest(self, request):  # type: ignore[no-untyped-def]
        self.calls.append(request)
        return IngestDocumentResponse(
            document_id=request.sha256,
            filename=request.filename,
            sha256=request.sha256,
            page_count=2,
            chunk_count=3,
            chunk_ids=("chunk-1", "chunk-2", "chunk-3"),
            config_hash="a" * 64,
        )

    def ingest_batch(self, requests, *, progress=None):  # type: ignore[no-untyped-def]
        if progress:
            progress("waiting_gpu", "Waiting for GPU", total=len(requests))
        return tuple(self.ingest(request) for request in requests)


def test_api_config_resolves_reviewed_pipeline_paths() -> None:
    config = ApiConfig.from_yaml(ROOT / "configs/api/default.yaml")
    assert config.default_mode == "two_pass"
    assert config.generation_config.name == "gemma_vllm_laptop.yaml"
    assert config.observability_config.name == "langfuse.yaml"
    paths = (path for path in config.model_dump().values() if isinstance(path, Path))
    assert all(path.is_absolute() for path in paths)


def test_fastapi_query_is_thin_and_returns_mandatory_provenance() -> None:
    service = FakeQueryService()
    client = TestClient(create_app(service=service))  # type: ignore[arg-type]
    assert client.get("/healthz").json() == {"status": "ok", "service": "findociq"}
    response = client.post(
        "/v1/query",
        json={"question": "What was revenue?", "question_id": "q1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "two_pass"
    assert payload["citations"][0] == {
        "document_id": "doc-1",
        "page_number": 2,
        "bbox": {"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0},
    }
    assert service.calls == [("What was revenue?", "two_pass", "q1")]


def test_fastapi_does_not_leak_backend_error_details() -> None:
    client = TestClient(create_app(service=FakeQueryService(fail=True)))  # type: ignore[arg-type]
    response = client.post("/v1/query", json={"question": "What was revenue?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "local inference pipeline unavailable"}
    assert "secret backend detail" not in response.text


def test_extract_returns_versioned_figures_with_mandatory_provenance() -> None:
    service = FakeQueryService()
    client = TestClient(create_app(service=service))  # type: ignore[arg-type]

    response = client.post(
        "/extract",
        json={"question": "Extract FY2025 revenue", "question_id": "case-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "contract_version": "1.0",
        "question": "Extract FY2025 revenue",
        "figures": [
            {
                "label": "Revenue",
                "value": "10",
                "unit": "INR crore",
                "period": "FY2025",
                "citation": {
                    "document_id": "doc-1",
                    "page_number": 2,
                    "bbox": {"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0},
                },
            }
        ],
        "notes": ["Consolidated revenue"],
    }
    assert service.calls == [("Extract FY2025 revenue", "two_pass", "case-1")]


def test_extract_rejects_unknown_request_fields() -> None:
    client = TestClient(create_app(service=FakeQueryService()))  # type: ignore[arg-type]

    response = client.post(
        "/extract",
        json={"question": "Extract revenue", "document_path": "private.pdf"},
    )

    assert response.status_code == 422


def test_document_ingestion_requires_token_and_returns_versioned_receipt() -> None:
    import base64
    import hashlib

    content = b"%PDF-1.7 synthetic"
    digest = hashlib.sha256(content).hexdigest()
    ingestion = FakeIngestionService()
    client = TestClient(
        create_app(
            service=FakeQueryService(),
            ingestion_service=ingestion,  # type: ignore[arg-type]
            ingest_token="local-ingestion-token-123456789",
        )
    )
    payload = {
        "filename": "borrower.pdf",
        "sha256": digest,
        "content_base64": base64.b64encode(content).decode(),
    }
    assert client.post("/v1/documents", json=payload).status_code == 401
    response = client.post(
        "/v1/documents",
        json=payload,
        headers={"X-FinDocIQ-Ingest-Token": "local-ingestion-token-123456789"},
    )
    assert response.status_code == 201
    assert response.json()["document_id"] == digest
    assert response.json()["chunk_count"] == 3
    assert len(ingestion.calls) == 1


def test_document_batch_uses_one_versioned_request() -> None:
    import base64
    import hashlib

    contents = (b"%PDF-1.7 first", b"%PDF-1.7 second")
    documents = [
        {
            "filename": f"borrower-{index}.pdf",
            "sha256": hashlib.sha256(content).hexdigest(),
            "content_base64": base64.b64encode(content).decode(),
        }
        for index, content in enumerate(contents, start=1)
    ]
    ingestion = FakeIngestionService()
    client = TestClient(
        create_app(
            service=FakeQueryService(),
            ingestion_service=ingestion,  # type: ignore[arg-type]
            ingest_token="local-ingestion-token-123456789",
        )
    )

    response = client.post(
        "/v1/document-batches",
        json={"documents": documents},
        headers={"X-FinDocIQ-Ingest-Token": "local-ingestion-token-123456789"},
    )

    assert response.status_code == 201
    assert len(response.json()["documents"]) == 2
    assert [request.filename for request in ingestion.calls] == [
        "borrower-1.pdf",
        "borrower-2.pdf",
    ]


def test_document_batch_exposes_sanitized_live_activity() -> None:
    import base64
    import hashlib

    content = b"%PDF-1.7 activity"
    digest = hashlib.sha256(content).hexdigest()
    token = "local-ingestion-token-123456789"
    client = TestClient(
        create_app(
            service=FakeQueryService(),
            ingestion_service=FakeIngestionService(),  # type: ignore[arg-type]
            ingest_token=token,
        )
    )
    response = client.post(
        "/v1/document-batches",
        json={
            "batch_id": "job-live-1",
            "documents": [
                {
                    "filename": "borrower.pdf",
                    "sha256": digest,
                    "content_base64": base64.b64encode(content).decode(),
                }
            ],
        },
        headers={"X-FinDocIQ-Ingest-Token": token},
    )
    assert response.status_code == 201

    activity = client.get(
        "/v1/document-batches/job-live-1/activity",
        headers={"X-FinDocIQ-Ingest-Token": token},
    )
    assert activity.status_code == 200
    assert activity.json()["status"] == "completed"
    assert activity.json()["events"][0]["stage"] == "waiting_gpu"
    assert "content_base64" not in activity.text
