"""FastAPI application with routes kept intentionally thin."""

import argparse
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import ValidationError

from findociq.api.activity import IngestionActivityRegistry
from findociq.api.ingestion import (
    DocumentIngestionError,
    DocumentIngestionService,
    build_ingestion_service,
)
from findociq.api.schema import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    IngestBatchRequest,
    IngestBatchResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    IngestionActivityResponse,
    ProductionExtractRequest,
    QueryRequest,
    QueryResponse,
    RetentionDeleteRequest,
    RetentionDeleteResponse,
)
from findociq.security.auth import ServiceJwtVerifier
from findociq.security.policy import ProductionGuardrailPolicy
from findociq.security.registry import (
    DocumentRecord,
    DocumentRegistry,
    InMemoryDocumentRegistry,
    PostgresSecurityStore,
)
from findociq.security.secrets import read_secret
from findociq.service import ApiConfig, FinDocIQService, build_service

LOGGER = logging.getLogger(__name__)


def create_app(
    service: FinDocIQService | None = None,
    *,
    config_path: Path = Path("configs/api/default.yaml"),
    ingestion_service: DocumentIngestionService | None = None,
    ingest_token: str | None = None,
    activity_registry: IngestionActivityRegistry | None = None,
    production_policy: ProductionGuardrailPolicy | None = None,
    service_jwt_verifier: ServiceJwtVerifier | None = None,
    document_registry: DocumentRegistry | None = None,
) -> FastAPI:
    production_enabled = os.getenv(
        "FINDOCIQ_PRODUCTION_GUARDRAILS_ENABLED", "false"
    ).casefold() in {"1", "true", "yes"}
    if production_enabled and production_policy is None:
        production_policy = ProductionGuardrailPolicy.from_yaml(
            os.getenv("FINDOCIQ_GUARDRAIL_POLICY", "configs/guardrails/production.yaml")
        )
    durable_security_store = None
    if production_enabled and document_registry is None:
        durable_security_store = PostgresSecurityStore(read_secret("FINDOCIQ_DATABASE_URL"))
        document_registry = durable_security_store
    if production_enabled and service_jwt_verifier is None:
        secret = _read_secret("FINDOCIQ_SERVICE_JWT_SECRET")
        service_jwt_verifier = ServiceJwtVerifier(
            secret, production_policy.service_auth, durable_security_store
        )
    app = FastAPI(title="FinDocIQ", version="0.2.0")
    app.state.query_service = service
    app.state.config_path = config_path
    app.state.ingestion_service = ingestion_service
    app.state.ingest_token = ingest_token
    app.state.activity_registry = activity_registry or IngestionActivityRegistry()
    app.state.production_enabled = production_enabled
    app.state.production_policy = production_policy
    app.state.service_jwt_verifier = service_jwt_verifier
    app.state.document_registry = document_registry or InMemoryDocumentRegistry()

    def authorize(
        authorization: str | None,
        correlation_id: str | None,
        *,
        required_role: str,
        application_id: str,
    ) -> None:
        if not app.state.production_enabled:
            return
        policy: ProductionGuardrailPolicy = app.state.production_policy
        if not correlation_id:
            raise HTTPException(400, "correlation ID required")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "valid service authorization required")
        try:
            app.state.service_jwt_verifier.verify(
                authorization.removeprefix("Bearer "),
                required_role=required_role,
                application_id=application_id,
            )
        except PermissionError as error:
            raise HTTPException(403, "authorization_scope_denied") from error
        except ValueError as error:
            raise HTTPException(401, "service_token_invalid") from error
        if policy.policy_hash != app.state.production_policy.policy_hash:
            raise HTTPException(503, "guardrail policy identity unavailable")

    def get_service(request: Request) -> FinDocIQService:
        current = request.app.state.query_service
        if current is None:
            current = build_service(ApiConfig.from_yaml(request.app.state.config_path))
            request.app.state.query_service = current
        return current

    def get_ingestion_service(request: Request) -> DocumentIngestionService:
        current = request.app.state.ingestion_service
        if current is None:
            root = os.environ.get("FINDOCIQ_DOCUMENT_ROOT")
            if not root:
                raise HTTPException(503, "document ingestion storage is not configured")
            api_config = ApiConfig.from_yaml(request.app.state.config_path)
            current = build_ingestion_service(
                storage_root=Path(root),
                ingestion_config=api_config.ingestion_config,
                index_config=api_config.index_config,
                retrieval_config=api_config.retrieval_config,
            )
            request.app.state.ingestion_service = current
        return current

    @app.get("/healthz", response_model=HealthResponse, response_model_exclude_none=True)
    def health() -> HealthResponse:
        return HealthResponse(
            policy_hash=(
                app.state.production_policy.policy_hash if app.state.production_enabled else None
            )
        )

    @app.post("/v1/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        query_service: Annotated[FinDocIQService, Depends(get_service)],
    ) -> QueryResponse:
        if app.state.production_enabled:
            raise HTTPException(404, "not found")
        mode = payload.mode or query_service.default_mode
        try:
            kwargs = {"mode": mode, "question_id": payload.question_id}
            if payload.document_ids:
                kwargs["document_ids"] = payload.document_ids
            result = query_service.query(payload.question, **kwargs)
        except (RuntimeError, OSError, ValueError) as error:
            LOGGER.exception("query inference failed")
            raise HTTPException(
                status_code=503, detail="local inference pipeline unavailable"
            ) from error
        return QueryResponse(
            mode=mode,
            answer=result.answer.answer,
            citations=result.answer.citations,
        )

    @app.post("/extract", response_model=ExtractResponse, response_model_exclude_none=True)
    def extract(
        payload: ExtractRequest | ProductionExtractRequest,
        query_service: Annotated[FinDocIQService, Depends(get_service)],
        authorization: Annotated[str | None, Header()] = None,
        x_correlation_id: Annotated[str | None, Header()] = None,
    ) -> ExtractResponse:
        if app.state.production_enabled:
            if not isinstance(payload, ProductionExtractRequest):
                raise HTTPException(422, "production extraction requires contract version 2.0")
            authorize(
                authorization,
                x_correlation_id,
                required_role="document_extract",
                application_id=payload.application_id,
            )
            policy = app.state.production_policy
            registry: DocumentRegistry = app.state.document_registry
            if not registry.owns(
                payload.application_id,
                payload.document_ids,
                policy_hash=policy.policy_hash,
            ):
                raise HTTPException(403, "documents do not belong to this application")
            try:
                question = "\n".join(
                    policy.extraction_question(metric_id) for metric_id in payload.metric_ids
                )
            except ValueError as error:
                raise HTTPException(422, "metric_not_allowed") from error
            question_id = payload.command_id
            document_ids = payload.document_ids
            application_id = payload.application_id
        else:
            if not isinstance(payload, ExtractRequest):
                raise HTTPException(422, "contract version 2.0 requires production guardrails")
            question = payload.question
            question_id = payload.question_id
            document_ids = payload.document_ids
            application_id = None
        try:
            kwargs = {"mode": "two_pass", "question_id": question_id}
            if document_ids:
                kwargs["document_ids"] = document_ids
            if application_id:
                kwargs["application_id"] = application_id
            result = query_service.query(question, **kwargs)
        except (RuntimeError, OSError, ValueError) as error:
            LOGGER.exception("structured extraction failed")
            raise HTTPException(
                status_code=503, detail="local inference pipeline unavailable"
            ) from error
        if result.extraction is None or not result.extraction.figures:
            raise HTTPException(status_code=502, detail="structured extraction unavailable")
        if app.state.production_enabled and any(
            figure.citation.document_id not in set(document_ids)
            for figure in result.extraction.figures
        ):
            raise HTTPException(502, "extraction returned foreign evidence")
        return ExtractResponse(
            contract_version="2.0" if app.state.production_enabled else "1.0",
            question=result.extraction.question,
            figures=result.extraction.figures,
            notes=result.extraction.notes,
            application_id=application_id,
            command_id=question_id if app.state.production_enabled else None,
            policy_hash=(
                app.state.production_policy.policy_hash if app.state.production_enabled else None
            ),
        )

    @app.post("/v1/documents", response_model=IngestDocumentResponse, status_code=201)
    def ingest_document(
        payload: IngestDocumentRequest,
        document_service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
        x_findociq_ingest_token: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
        x_correlation_id: Annotated[str | None, Header()] = None,
    ) -> IngestDocumentResponse:
        if app.state.production_enabled:
            if payload.contract_version != "2.0" or not payload.application_id:
                raise HTTPException(422, "production ingestion requires application scope")
            authorize(
                authorization,
                x_correlation_id,
                required_role="document_ingest",
                application_id=payload.application_id,
            )
        else:
            expected = app.state.ingest_token or os.getenv("FINDOCIQ_INGEST_TOKEN")
            if not expected or not x_findociq_ingest_token or not secrets.compare_digest(
                expected, x_findociq_ingest_token
            ):
                raise HTTPException(401, "valid ingestion token required")
        try:
            result = document_service.ingest(payload)
            return _register_document(app, result)
        except (DocumentIngestionError, ValidationError) as error:
            detail = _safe_document_code(error) if app.state.production_enabled else str(error)
            raise HTTPException(422, detail) from error
        except (RuntimeError, OSError) as error:
            raise HTTPException(503, "document ingestion pipeline unavailable") from error

    @app.post("/v1/document-batches", response_model=IngestBatchResponse, status_code=201)
    def ingest_document_batch(
        payload: IngestBatchRequest,
        document_service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
        x_findociq_ingest_token: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
        x_correlation_id: Annotated[str | None, Header()] = None,
    ) -> IngestBatchResponse:
        application_id = payload.documents[0].application_id
        if app.state.production_enabled:
            if payload.contract_version != "2.0" or not application_id:
                raise HTTPException(422, "production ingestion requires application scope")
            authorize(
                authorization,
                x_correlation_id,
                required_role="document_ingest",
                application_id=application_id,
            )
        else:
            expected = app.state.ingest_token or os.getenv("FINDOCIQ_INGEST_TOKEN")
            if not expected or not x_findociq_ingest_token or not secrets.compare_digest(
                expected, x_findociq_ingest_token
            ):
                raise HTTPException(401, "valid ingestion token required")
        registry: IngestionActivityRegistry = app.state.activity_registry
        if payload.batch_id:
            registry.start(payload.batch_id, application_id)

        def report(stage: str, message: str, **details: object) -> None:
            if payload.batch_id:
                registry.report(payload.batch_id, stage, message, **details)

        try:
            documents = tuple(
                _register_document(app, document)
                for document in document_service.ingest_batch(
                    payload.documents, progress=report
                )
            )
            if payload.batch_id:
                registry.finish(payload.batch_id, "completed")
            return IngestBatchResponse(
                contract_version=payload.contract_version, documents=documents
            )
        except (DocumentIngestionError, ValidationError) as error:
            if payload.batch_id:
                registry.finish(payload.batch_id, "failed")
            detail = _safe_document_code(error) if app.state.production_enabled else str(error)
            raise HTTPException(422, detail) from error
        except (RuntimeError, OSError) as error:
            if payload.batch_id:
                registry.finish(payload.batch_id, "failed")
            raise HTTPException(503, "document ingestion pipeline unavailable") from error
        except Exception:
            if payload.batch_id:
                registry.finish(payload.batch_id, "failed")
            raise

    @app.get(
        "/v1/document-batches/{batch_id}/activity",
        response_model=IngestionActivityResponse,
    )
    def document_batch_activity(
        batch_id: str,
        after: int = 0,
        x_findociq_ingest_token: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
        x_correlation_id: Annotated[str | None, Header()] = None,
    ) -> IngestionActivityResponse:
        if app.state.production_enabled:
            application_id = app.state.activity_registry.application_id(batch_id)
            if not application_id:
                raise HTTPException(404, "document batch activity not found")
            authorize(
                authorization,
                x_correlation_id,
                required_role="activity_read",
                application_id=application_id,
            )
        else:
            expected = app.state.ingest_token or os.getenv("FINDOCIQ_INGEST_TOKEN")
            if not expected or not x_findociq_ingest_token or not secrets.compare_digest(
                expected, x_findociq_ingest_token
            ):
                raise HTTPException(401, "valid ingestion token required")
        snapshot = app.state.activity_registry.snapshot(batch_id, after=max(0, after))
        if snapshot is None:
            raise HTTPException(404, "document batch activity not found")
        return snapshot

    @app.post(
        "/v1/applications/{application_id}/retention-delete",
        response_model=RetentionDeleteResponse,
    )
    def retention_delete(
        application_id: str,
        payload: RetentionDeleteRequest,
        query_service: Annotated[FinDocIQService, Depends(get_service)],
        document_service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
        authorization: Annotated[str | None, Header()] = None,
        x_correlation_id: Annotated[str | None, Header()] = None,
    ) -> RetentionDeleteResponse:
        if not app.state.production_enabled:
            raise HTTPException(404, "not found")
        authorize(
            authorization,
            x_correlation_id,
            required_role="document_delete",
            application_id=application_id,
        )
        policy: ProductionGuardrailPolicy = app.state.production_policy
        registry: DocumentRegistry = app.state.document_registry
        document_ids = registry.delete_application(application_id, policy.policy_hash)
        store = getattr(query_service.retrieval, "store", None)
        if store is None or not hasattr(store, "delete_application"):
            raise HTTPException(503, "document deletion store unavailable")
        store.delete_application(application_id)
        if document_service.encrypted_store is not None:
            document_service.encrypted_store.delete_application(application_id)
        canonical = "|".join(
            (application_id, payload.command_id, *document_ids, policy.policy_hash)
        )
        return RetentionDeleteResponse(
            application_id=application_id,
            deleted_document_ids=document_ids,
            receipt_sha256=sha256(canonical.encode()).hexdigest(),
            policy_hash=policy.policy_hash,
        )

    return app


def _read_secret(name: str) -> str:
    return read_secret(name)


def _register_document(app: FastAPI, response: IngestDocumentResponse) -> IngestDocumentResponse:
    if not app.state.production_enabled:
        return response
    if not response.application_id or not response.policy_hash:
        raise RuntimeError("production ingestion returned an unscoped receipt")
    registry: DocumentRegistry = app.state.document_registry
    policy: ProductionGuardrailPolicy = app.state.production_policy
    registry.register(
        DocumentRecord(
            application_id=response.application_id,
            document_id=response.document_id,
            sha256=response.sha256,
            policy_hash=response.policy_hash,
            retained_until=datetime.now(UTC)
            + timedelta(days=policy.retention.terminal_days),
        )
    )
    query_service: FinDocIQService | None = app.state.query_service
    if query_service is not None:
        query_service.retrieval.invalidate_application(response.application_id)
    canonical = "|".join(
        (
            response.application_id,
            response.document_id,
            response.sha256,
            response.policy_hash,
        )
    )
    return response.model_copy(
        update={"ownership_receipt_sha256": sha256(canonical.encode()).hexdigest()}
    )


def _safe_document_code(error: Exception) -> str:
    code = str(error)
    allowed = {
        "active_pdf_content",
        "document_prompt_injection_risk",
        "document_parse_timeout",
        "embedded_file",
        "encrypted_pdf",
        "malformed_pdf",
        "malware_detected",
        "malware_scan_inconclusive",
        "malware_scanner_unavailable",
        "malware_scanner_version_unavailable",
    }
    return code if code in allowed else "document_rejected"


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8989)
    parser.add_argument("--config", type=Path, default=Path("configs/api/default.yaml"))
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("API serving requires `uv sync --extra api`") from error
    uvicorn.run(create_app(config_path=args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
