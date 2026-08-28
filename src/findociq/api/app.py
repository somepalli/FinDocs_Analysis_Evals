"""FastAPI application with routes kept intentionally thin."""

import argparse
import logging
import os
import secrets
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
    QueryRequest,
    QueryResponse,
)
from findociq.service import ApiConfig, FinDocIQService, build_service

LOGGER = logging.getLogger(__name__)


def create_app(
    service: FinDocIQService | None = None,
    *,
    config_path: Path = Path("configs/api/default.yaml"),
    ingestion_service: DocumentIngestionService | None = None,
    ingest_token: str | None = None,
    activity_registry: IngestionActivityRegistry | None = None,
) -> FastAPI:
    app = FastAPI(title="FinDocIQ", version="0.2.0")
    app.state.query_service = service
    app.state.config_path = config_path
    app.state.ingestion_service = ingestion_service
    app.state.ingest_token = ingest_token
    app.state.activity_registry = activity_registry or IngestionActivityRegistry()

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

    @app.get("/healthz", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/v1/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        query_service: Annotated[FinDocIQService, Depends(get_service)],
    ) -> QueryResponse:
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

    @app.post("/extract", response_model=ExtractResponse)
    def extract(
        payload: ExtractRequest,
        query_service: Annotated[FinDocIQService, Depends(get_service)],
    ) -> ExtractResponse:
        try:
            kwargs = {"mode": "two_pass", "question_id": payload.question_id}
            if payload.document_ids:
                kwargs["document_ids"] = payload.document_ids
            result = query_service.query(payload.question, **kwargs)
        except (RuntimeError, OSError, ValueError) as error:
            LOGGER.exception("structured extraction failed")
            raise HTTPException(
                status_code=503, detail="local inference pipeline unavailable"
            ) from error
        if result.extraction is None or not result.extraction.figures:
            raise HTTPException(status_code=502, detail="structured extraction unavailable")
        return ExtractResponse(
            question=result.extraction.question,
            figures=result.extraction.figures,
            notes=result.extraction.notes,
        )

    @app.post("/v1/documents", response_model=IngestDocumentResponse, status_code=201)
    def ingest_document(
        payload: IngestDocumentRequest,
        document_service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
        x_findociq_ingest_token: Annotated[str | None, Header()] = None,
    ) -> IngestDocumentResponse:
        expected = app.state.ingest_token or os.getenv("FINDOCIQ_INGEST_TOKEN")
        if not expected or not x_findociq_ingest_token or not secrets.compare_digest(
            expected, x_findociq_ingest_token
        ):
            raise HTTPException(401, "valid ingestion token required")
        try:
            return document_service.ingest(payload)
        except (DocumentIngestionError, ValidationError) as error:
            raise HTTPException(422, str(error)) from error
        except (RuntimeError, OSError) as error:
            raise HTTPException(503, "document ingestion pipeline unavailable") from error

    @app.post("/v1/document-batches", response_model=IngestBatchResponse, status_code=201)
    def ingest_document_batch(
        payload: IngestBatchRequest,
        document_service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
        x_findociq_ingest_token: Annotated[str | None, Header()] = None,
    ) -> IngestBatchResponse:
        expected = app.state.ingest_token or os.getenv("FINDOCIQ_INGEST_TOKEN")
        if not expected or not x_findociq_ingest_token or not secrets.compare_digest(
            expected, x_findociq_ingest_token
        ):
            raise HTTPException(401, "valid ingestion token required")
        registry: IngestionActivityRegistry = app.state.activity_registry
        if payload.batch_id:
            registry.start(payload.batch_id)

        def report(stage: str, message: str, **details: object) -> None:
            if payload.batch_id:
                registry.report(payload.batch_id, stage, message, **details)

        try:
            documents = document_service.ingest_batch(payload.documents, progress=report)
            if payload.batch_id:
                registry.finish(payload.batch_id, "completed")
            return IngestBatchResponse(documents=documents)
        except (DocumentIngestionError, ValidationError) as error:
            if payload.batch_id:
                registry.finish(payload.batch_id, "failed")
            raise HTTPException(422, str(error)) from error
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
    ) -> IngestionActivityResponse:
        expected = app.state.ingest_token or os.getenv("FINDOCIQ_INGEST_TOKEN")
        if not expected or not x_findociq_ingest_token or not secrets.compare_digest(
            expected, x_findociq_ingest_token
        ):
            raise HTTPException(401, "valid ingestion token required")
        snapshot = app.state.activity_registry.snapshot(batch_id, after=max(0, after))
        if snapshot is None:
            raise HTTPException(404, "document batch activity not found")
        return snapshot

    return app


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
