"""Read-only private acceptance assessment using Qdrant HTTP, without model calls."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import TypeAdapter

from findociq.ingest.schema import Chunk
from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, EvidencePolicy
from findociq.reason.interpretations import interpretation_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--application-id", required=True)
    parser.add_argument("--development-unscoped", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:6999")
    args = parser.parse_args()
    document_ids = tuple(
        d["document_id"] for d in json.loads(args.manifest.read_text())["documents"]
    )
    conditions = [{"key": "chunk.provenance[].document_id", "match": {"any": list(document_ids)}}]
    if not args.development_unscoped:
        conditions.append(
            {"key": "chunk.metadata.application_id", "match": {"value": args.application_id}}
        )
    chunks, offset = [], None
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    while True:
        body = {
            "filter": {"must": conditions},
            "limit": 256,
            "with_vectors": False,
            "with_payload": True,
            "offset": offset,
        }
        request = Request(
            args.url + "/collections/findociq_chunks/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            result = json.load(response)["result"]
        for point in result["points"]:
            chunk = TypeAdapter(Chunk).validate_python(point["payload"]["chunk"])
            if any(p.document_id not in document_ids for p in chunk.provenance):
                raise RuntimeError("foreign_evidence")
            if (
                not args.development_unscoped
                and chunk.metadata.get("application_id") != args.application_id
            ):
                raise RuntimeError("foreign_evidence")
            chunks.append(chunk)
        if len(chunks) > policy.max_chunks:
            raise RuntimeError("inventory_limit")
        offset = result.get("next_page_offset")
        if offset is None:
            break
    if {p.document_id for c in chunks for p in c.provenance} != set(document_ids):
        raise RuntimeError("document_missing")
    gate = BorrowerEvidenceGate(policy)
    outcomes = {}
    for metric in policy.metrics:
        try:
            figure = gate.extract(metric, tuple(chunks))
            outcomes[metric] = {
                "status": "supported",
                "citation": figure.citation.model_dump(),
                "period": figure.period,
                "basis": figure.evidence_validation.basis,
            }
        except EvidenceInsufficient as error:
            outcomes[metric] = {"status": "unresolved", "code": error.code}
    basis_pages = {}
    for document in document_ids:
        basis_pages[document] = sorted(
            {
                p.page_number
                for c in chunks
                for p in c.provenance
                if p.document_id == document
                and any(term in c.text.lower() for term in ("standalone", "consolidated"))
            }
        )
    print(
        json.dumps(
            {
                "documents": len(document_ids),
                "chunks": len(chunks),
                "unapproved_reporting_scope_options": sum(
                    option.authority == "reporting_scope"
                    for option in interpretation_options(tuple(chunks), policy)
                ),
                "basis_mention_pages": basis_pages,
                "metrics": outcomes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"status": "unavailable", "code": "assessment_failed"}))
        raise SystemExit(2) from None
