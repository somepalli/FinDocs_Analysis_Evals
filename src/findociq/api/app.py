"""FastAPI application with routes kept intentionally thin."""

import argparse
import os
import secrets
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import ValidationError

from findociq.api.ingestion import (
    DocumentIngestionError,
    DocumentIngestionService,
    build_ingestion_service,
)
from findociq.api.schema import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    QueryRequest,
    QueryResponse,
)
from findociq.service import ApiConfig, FinDocIQService, build_service


def create_app(
    service: FinDocIQService | None = None,
    *,
    config_path: Path = Path("configs/api/default.yaml"),
    ingestion_service: DocumentIngestionService | None = None,
    ingest_token: str | None = None,
) -> FastAPI:
    app = FastAPI(title="FinDocIQ", version="0.1.1")
    app.state.query_service = service
    app.state.config_path = config_path
    app.state.ingestion_service = ingestion_service
    app.state.ingest_token = ingest_token

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
            current = build_ingestion_service(
                storage_root=Path(root),
                ingestion_config=Path("configs/ingestion/default.yaml"),
                index_config=Path("configs/index/default.yaml"),
                retrieval_config=Path("configs/retrieval/hybrid_rerank.yaml"),
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
        except (RuntimeError, OSError) as error:
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
        except (RuntimeError, OSError) as error:
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
