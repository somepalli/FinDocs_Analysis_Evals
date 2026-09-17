from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_evidence_gate import statement

from findociq.api.app import create_app
from findociq.index.store import QdrantStore, QdrantStoreConfig
from findociq.observability.recorder import InMemoryRecorder, TraceObserver
from findociq.reason.evidence_gate import EvidenceInsufficient, EvidencePolicy
from findociq.security.policy import ProductionGuardrailPolicy
from findociq.service import FinDocIQService


def store_with_pages(pages):
    store = QdrantStore(QdrantStoreConfig())
    calls = []

    def scroll(**kwargs):
        calls.append(kwargs)
        return pages[len(calls) - 1]

    store._dependencies = lambda: (SimpleNamespace(scroll=scroll), None)
    store._document_filter = lambda _, docs, app: {"documents": docs, "application": app}
    return store, calls


def record(doc="financials", app="APP-1"):
    source = statement().model_copy(update={"metadata": {"application_id": app}})
    provenance = source.provenance[0].model_copy(update={"document_id": doc})
    source = source.model_copy(update={"provenance": (provenance,)})
    return SimpleNamespace(payload={"chunk": source.model_dump(mode="json")})


def test_inventory_scroll_enforces_scope_on_every_page():
    store, calls = store_with_pages([([record()], "next"), ([record()], None)])
    assert len(store.scoped_chunks(("financials",), "APP-1")) == 2
    assert all(
        c["scroll_filter"] == {"documents": ("financials",), "application": "APP-1"} for c in calls
    )
    assert calls[1]["offset"] == "next"


@pytest.mark.parametrize("doc,app", [("foreign", "APP-1"), ("financials", "APP-2")])
def test_inventory_rejects_foreign_returned_chunks(doc, app):
    store, _ = store_with_pages([([record(doc, app)], None)])
    with pytest.raises(PermissionError):
        store.scoped_chunks(("financials",), "APP-1")


def test_inventory_does_not_silently_truncate_or_ignore_missing_documents():
    store, _ = store_with_pages([([record(), record()], None)])
    with pytest.raises(ValueError, match="inventory_limit"):
        store.scoped_chunks(("financials",), max_chunks=1)
    store, _ = store_with_pages([([record()], None)])
    with pytest.raises(ValueError, match="document_missing"):
        store.scoped_chunks(("financials", "missing"))
    with pytest.raises(ValueError, match="scope_missing"):
        store.scoped_chunks(())


def test_metric_request_uses_scoped_evidence_without_generation_and_trace_leak():
    store, _ = store_with_pages([([record()], None)])
    recorder = InMemoryRecorder()
    service = FinDocIQService(
        SimpleNamespace(store=store),
        {},
        TraceObserver(recorder),
        evidence_policy=EvidencePolicy.load(Path("configs/evidence/borrower.yaml")),
    )
    response = TestClient(create_app(service=service)).post(
        "/extract",
        json={
            "question": "Extract revenue",
            "metric_id": "annual_revenue_crore",
            "document_ids": ["financials"],
        },
    )
    assert response.status_code == 200
    assert response.json()["figures"][0]["value"] == "12.00"
    assert response.json()["figures"][0]["evidence_validation"]["source_value"] == "1200"
    assert [e.stage for e in recorder.events] == ["evidence.selection"]
    serialized = "".join(e.model_dump_json() for e in recorder.events)
    assert "Example Manufacturing" not in serialized
    assert "1200" not in serialized
    assert "Revenue from operations" not in serialized


def test_metric_failure_is_safe_422_not_model_outage():
    class Missing:
        def extract_metrics(self, *args, **kwargs):
            raise EvidenceInsufficient("evidence_unit_ambiguous")

    response = TestClient(create_app(service=Missing())).post(
        "/extract",
        json={
            "question": "Extract revenue",
            "metric_id": "annual_revenue_crore",
            "document_ids": ["financials"],
        },
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "evidence_unit_ambiguous"}


def test_production_ownership_denial_has_terminal_evidence_code(monkeypatch):
    monkeypatch.setenv("FINDOCIQ_PRODUCTION_GUARDRAILS_ENABLED", "true")
    app = create_app(
        service=SimpleNamespace(),
        production_policy=ProductionGuardrailPolicy.from_yaml("configs/guardrails/production.yaml"),
        service_jwt_verifier=SimpleNamespace(verify=lambda *args, **kwargs: None),
        document_registry=SimpleNamespace(owns=lambda *args, **kwargs: False),
    )
    response = TestClient(app).post(
        "/extract",
        json={
            "contract_version": "2.0",
            "application_id": "APP-1",
            "document_ids": ["foreign"],
            "metric_ids": ["annual_revenue_crore"],
            "command_id": "command-1",
        },
        headers={"Authorization": "Bearer synthetic", "X-Correlation-ID": "test-1"},
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "evidence_scope_violation"}
