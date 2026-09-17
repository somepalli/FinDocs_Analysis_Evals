# Borrower evidence selection

FunderMatch's explicit metric extraction is separate from general question answering and benchmark
evaluation. All uploaded PDFs may need safe parsing to discover their contents; they are **not all
accepted as financial evidence**, and uploaded borrower data is never automatically added to evals.

`configs/evidence/borrower.yaml` is the versioned, hashed policy. The metric path reads a bounded,
document-scoped chunk inventory from Qdrant, checks document type and borrower identity, and selects
an exact labelled source field. Application scope is additionally mandatory in production.
The store rejects foreign returned chunks, missing documents, and truncated inventories.

| Source | Permitted use |
|---|---|
| Audited financial statements | Revenue, PAT and explicitly reported financial ratios |
| Registration/agreement | Identity and supported operating facts |
| Payroll/employee proof | Explicit employee count |
| Matching utility bill | Address-to-region mapping only |
| Valuation report | Explicit dated collateral coverage |
| No-dues certificate | Not a substitute for revenue, profitability or DSCR |

Financial values require an explicit reporting-year column, accounting basis where applicable,
and monetary units. Decimal conversions to INR crore retain the original cell, factor, chunk ID,
page/bbox and policy hash in `evidence_validation`. Percentages require explicit percentage notation.
Newer statements without the requested metric cannot silently fall back to an older reported year.
Conflicting amounts and mixed borrower identities abstain. Location uses the policy's state-to-region
mapping; unsupported or conflicting addresses abstain rather than invent a funder-policy region.

Recognized tables and label/value rows are handled deterministically; this metric path does not call
Gemma. Unknown layouts, absent metrics and ambiguous context return safe HTTP 422 `evidence_*` codes.
FunderMatch routes those to `needs_attention`, and independently checks receipts and cross-metric
consistency. General development `/v1/query` and free-text `/extract` without `metric_id` retain their
existing retrieval/reasoning path. Production v2 metric extraction always uses this gate.

Operators can run a read-only readiness assessment against an ingested DocumentArtifact manifest:

```powershell
uv run --extra retrieval python scripts/assess_borrower_evidence.py `
  --manifest <document_processing.json> --application-id <application-id>
```

Only document IDs/types, counts, policy hash and metric status codes are printed. The explicit
`--development-unscoped` switch is for older local ingestion only; never use it for production.
It is an operator diagnostic, not an unauthenticated public API.

Limitations: classification does not authenticate an auditor, signature, document, or issuer.
An internally consistent wrong OCR cell still needs a PDF spot-check. Ratios are not fabricated
from absent inputs. The parser currently accepts conservative pipe-table and labelled-field layouts;
unrecognized evidence requires review or a supported clearer source. These checks do not claim
real-service extraction accuracy or replace the existing held-out evals.
