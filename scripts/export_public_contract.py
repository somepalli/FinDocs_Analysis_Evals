"""Export or verify FinDocIQ's versioned public HTTP contract bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from findociq.api.schema import (
    DocumentSetExtractRequest,
    DocumentSetRequest,
    DocumentSetResponse,
    DocumentSetValidationRequest,
    DocumentSetValidationResponse,
    EvidenceAssessmentResponse,
    ExtractRequest,
    ExtractResponse,
    IngestBatchRequest,
    IngestBatchResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    IngestionActivityResponse,
    ProductionExtractRequest,
    RetentionDeleteRequest,
    RetentionDeleteResponse,
)

CONTRACT_PATH = Path("contracts/findociq-public-http-contract.json")
MODELS: dict[str, type[BaseModel]] = {
    model.__name__: model
    for model in (
        DocumentSetRequest,
        DocumentSetResponse,
        DocumentSetValidationRequest,
        DocumentSetValidationResponse,
        DocumentSetExtractRequest,
        EvidenceAssessmentResponse,
        ExtractRequest,
        ProductionExtractRequest,
        ExtractResponse,
        IngestDocumentRequest,
        IngestDocumentResponse,
        IngestBatchRequest,
        IngestBatchResponse,
        IngestionActivityResponse,
        RetentionDeleteRequest,
        RetentionDeleteResponse,
    )
}


def normalize_schema(value: Any) -> Any:
    """Remove presentation-only fields while preserving validation semantics."""
    if isinstance(value, dict):
        return {
            key: normalize_schema(item)
            for key, item in sorted(value.items())
            if key not in {"description", "title"}
        }
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    return value


def build_contract() -> dict[str, object]:
    return {
        "bundle_version": "1.0",
        "service": "findociq",
        "supported_contract_versions": ["1.0", "2.0", "2.1"],
        "endpoints": {
            "/v1/document-sets/validate": {
                "request": "DocumentSetValidationRequest",
                "response": "DocumentSetValidationResponse",
            },
            "/v1/document-sets": {
                "request": "DocumentSetRequest",
                "response": "DocumentSetResponse",
            },
            "/v1/evidence-assessments": {
                "request": "DocumentSetExtractRequest",
                "response": "EvidenceAssessmentResponse",
            },
            "/extract": {
                "development_request": "ExtractRequest",
                "production_request": "ProductionExtractRequest",
                "document_set_request": "DocumentSetExtractRequest",
                "response": "ExtractResponse",
            },
            "/v1/documents": {
                "request": "IngestDocumentRequest",
                "response": "IngestDocumentResponse",
            },
            "/v1/document-batches": {
                "request": "IngestBatchRequest",
                "response": "IngestBatchResponse",
            },
            "/v1/ingestion-activity/{batch_id}": {"response": "IngestionActivityResponse"},
            "/v1/applications/{application_id}/documents": {
                "request": "RetentionDeleteRequest",
                "response": "RetentionDeleteResponse",
            },
        },
        "schemas": {
            name: normalize_schema(model.model_json_schema())
            for name, model in sorted(MODELS.items())
        },
    }


def encoded_contract() -> str:
    return json.dumps(build_contract(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = encoded_contract()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"public contract snapshot is stale; run: uv run python {__file__}")
        print(f"public contract snapshot is current: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
