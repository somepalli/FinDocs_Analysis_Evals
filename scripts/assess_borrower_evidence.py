"""Read-only, content-safe evidence readiness report; no inference or workflow writes."""

import argparse
import json
from pathlib import Path

from findociq.index.store import QdrantStore, QdrantStoreConfig
from findociq.reason.evidence_gate import (
    BorrowerEvidenceGate,
    EvidenceInsufficient,
    EvidencePolicy,
    profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="DocumentArtifact JSON")
    parser.add_argument("--application-id", required=True)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6999")
    parser.add_argument("--collection", default="findociq_chunks")
    parser.add_argument("--policy", type=Path, default=Path("configs/evidence/borrower.yaml"))
    parser.add_argument(
        "--development-unscoped",
        action="store_true",
        help="Explicit dev-only compatibility for older unscoped ingestion",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = tuple(d["document_id"] for d in manifest["documents"])
    policy = EvidencePolicy.load(args.policy)
    chunks = QdrantStore(
        QdrantStoreConfig(url=args.qdrant_url, collection=args.collection)
    ).scoped_chunks(
        documents,
        None if args.development_unscoped else args.application_id,
        max_chunks=policy.max_chunks,
    )
    gate = BorrowerEvidenceGate(policy)
    statuses = {}
    for metric in policy.metrics:
        try:
            gate.extract(metric, chunks)
            statuses[metric] = "supported"
        except EvidenceInsufficient as error:
            statuses[metric] = error.code
    inventory = [
        {
            "document_id": document,
            "types": sorted(
                profile(
                    document,
                    tuple(
                        c for c in chunks if any(p.document_id == document for p in c.provenance)
                    ),
                    policy,
                ).types
            ),
        }
        for document in documents
    ]
    print(
        json.dumps(
            {
                "application_id": args.application_id,
                "policy_hash": policy.policy_hash,
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "inventory": inventory,
                "metrics": statuses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A malformed chunk validation error may contain source text. Never print it.
        print(json.dumps({"status": "unavailable", "error_code": "evidence_assessment_failed"}))
        raise SystemExit(2) from None
