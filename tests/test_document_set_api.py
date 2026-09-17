from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from test_basis_context import page, sources

from findociq.api.app import create_app
from findociq.api.schema import IngestDocumentResponse
from findociq.reason.evidence_gate import EvidencePolicy
from findociq.service import FinDocIQService


def test_37_document_http_flow_and_full_assessment(tmp_path, monkeypatch):
    monkeypatch.setenv("FINDOCIQ_DOCUMENT_ROOT", str(tmp_path))
    monkeypatch.setenv("FINDOCIQ_PRODUCTION_GUARDRAILS_ENABLED", "false")
    ids = tuple(["report"] + [f"proof-{i}" for i in range(36)])
    chunks = tuple(
        c.model_copy(update={"metadata": {"application_id": "APP-A"}})
        for c in sources() + tuple(page("Supporting document", document=d) for d in ids[1:])
    )

    class Store:
        def scoped_chunks(self, document_ids, application_id, *, max_chunks):
            assert set(document_ids) == set(ids)
            assert application_id == "APP-A"
            return chunks

    class Ingestion:
        def ingest_batch(self, documents, *, progress):
            return tuple(
                IngestDocumentResponse(
                    document_id=d,
                    filename=f"{d}.pdf",
                    sha256="a" * 64,
                    page_count=2,
                    chunk_count=1,
                    chunk_ids=(d,),
                    config_hash="test",
                    application_id="APP-A",
                )
                for d in ids
            )

    service = FinDocIQService(
        SimpleNamespace(store=Store()),
        {},
        evidence_policy=EvidencePolicy.load(Path("configs/evidence/borrower.yaml")),
    )
    app = create_app(service, ingestion_service=Ingestion(), ingest_token="test-only-token")
    client = TestClient(app)
    headers = {"X-FinDocIQ-Ingest-Token": "test-only-token"}
    batch = {
        "documents": [
            {
                "filename": f"{d}.pdf",
                "sha256": "a" * 64,
                "content_base64": "JVBERi0xLjQ=",
                "application_id": "APP-A",
            }
            for d in ids
        ]
    }
    response = client.post("/v1/document-batches", json=batch, headers=headers)
    assert response.status_code == 201, response.text
    set_id = response.json()["document_set_id"]
    request = {
        "contract_version": "2.1",
        "application_id": "APP-A",
        "document_set_id": set_id,
        "metric_ids": ["annual_revenue_crore"],
        "command_id": "synthetic-command",
    }
    extraction = client.post("/extract", json=request, headers=headers)
    assert extraction.status_code == 200, extraction.text
    assert extraction.json()["figures"][0]["value"] == "12.00"
    assert extraction.json()["contract_version"] == "2.1"
    ignored_interpretation = client.post(
        "/extract", json={**request, "interpretation_command_id": "a" * 36}, headers=headers
    )
    assert ignored_interpretation.status_code == 422
    assert ignored_interpretation.json()["detail"] == "use_evidence_assessments_for_interpretations"
    request["metric_ids"] = ["annual_revenue_crore", "dscr", "employee_count"]
    assessment = client.post("/v1/evidence-assessments", json=request, headers=headers)
    assert assessment.status_code == 200, assessment.text
    assert [f["status"] for f in assessment.json()["fields"]] == ["supported", "missing", "missing"]
    assert assessment.json()["classifications"]
    assert client.post("/extract", json=request).status_code == 401
    request["application_id"] = "APP-B"
    assert client.post("/extract", json=request, headers=headers).status_code == 403
    # Document sets survive API recreation, including the authoritative local registry.
    restarted = TestClient(create_app(service, ingest_token="test-only-token"))
    request["application_id"] = "APP-A"
    assert (
        restarted.post("/v1/evidence-assessments", json=request, headers=headers).status_code == 200
    )
    validation_request = {"application_id": "APP-A", "document_set_id": set_id}
    validation = restarted.post(
        "/v1/document-sets/validate", json=validation_request, headers=headers
    )
    assert validation.status_code == 200
    assert validation.json()["document_count"] == 37
    assert validation.json()["evidence_policy_hash"] == service.evidence_policy.policy_hash
    assert restarted.post("/v1/document-sets/validate", json=validation_request).status_code == 401
    assert (
        restarted.post(
            "/v1/document-sets/validate",
            json={**validation_request, "application_id": "APP-B"},
            headers=headers,
        ).status_code
        == 403
    )
    restarted.app.state.document_registry.delete_application(
        "APP-A", validation.json()["policy_hash"]
    )
    assert (
        restarted.post(
            "/v1/document-sets/validate", json=validation_request, headers=headers
        ).status_code
        == 403
    )
