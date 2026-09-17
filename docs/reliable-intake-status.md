# Reliable intake: implementation and qualification status

This is deployed for approved **development-only synthetic qualification**, not
a production release. Existing borrower applications and documents remain untouched.

## Implemented

- Contract 2.1 document-set extraction, alongside unchanged 1.0/2.0 request limits.
  Membership is immutable/content-addressed, application-scoped, and checked against
  current ownership on every access. Production uses PostgreSQL; explicit local
  development persists its ownership and sets in SQLite under the document root.
- Bounded Qdrant inventory reads preserve all members; missing or oversized inventories
  fail rather than truncating. New scoped batch responses include the document-set ID.
- Per-field evidence assessment and page-range content classification, with provenance.
  Classification is deterministic keyword evidence, not document authentication or
  a trained layout classifier. Unknown/ambiguous pages stay explicitly unresolved.
- Cross-page report declarations require matching reporting period. There is no
  default-to-standalone rule. Mixed/unlinked declarations remain blocked.
- Separate table provenance prevents surrounding paragraphs from supplying a figure's
  bounding box. Existing multi-provenance chunks without that distinction remain
  conservative; this change does not rewrite previously indexed borrower evidence.
- FunderMatch forwards backend activity, distinguishes missing progress from backend
  unreachability, and exposes a masked, read-only full-field evidence review panel.
- Invalid extraction contracts are not classified as retryable dependency outages.
- Upload capability discovery and preflight limits, retaining larger agent uploads
  while bounding the legacy direct intake to its existing 20-document contract.

## Verification implemented

- Recovery continuation: FinDocIQ exposes an authenticated document-set validation
  operation that rechecks ownership and current policy without OCR/extraction.
  FunderMatch reuses supported fields only for the same immutable set and evidence
  policy, then requests unresolved fields only. Different metric batches have
  distinct stable command IDs. Classifier rules are pinned into evidence policy v3
  and its hash; changing them invalidates cached findings.
- Supplement staging can persist an encrypted, immutable candidate manifest, reject
  stale request hashes, detect repeated-command payload conflicts and deduplicate
  content without changing active inputs or existing PDFs. This is a storage
  primitive now has a FunderMatch reviewer-only API and explicit-resume UI behind
  `FUNDERMATCH_EVIDENCE_REPAIR_ENABLED=false`. PostgreSQL commits the repair intent
  with its human audit; operation fences serialize activation and resume. The
  document worker ingests only new hashes, persists a batch receipt, and creates
  a combined immutable set. FunderMatch migration 007 is required before rollout.
- A controlled-service graph recovery test reaches human review after a missing
  DSCR assessment is resolved, without repeating ingestion or the other twelve
  completed field calls. This uses test doubles and in-memory graph/workflow
  stores, not PostgreSQL restart or real-borrower acceptance.

- Five invented PDFs (registration, GST, LLP agreement, two annual statements) pass
  actual digital PDF parsing, table extraction, chunking, classification and all
  13 field assessments. These are CPU digital-fast-path tests, **not GPU OCR tests**.
- A 37-document HTTP test covers ingestion receipts, document-set extraction,
  full assessment, foreign-owner rejection and local store reopening. Ingestion
  and vector storage are controlled doubles, **not a real-service end-to-end run**.
- 5/20/21/37-member immutable-set persistence, deduplication and revoked-ownership tests.
- Basis context, mixed proofs, activity cancellation/outage/deduplication, actual
  JavaScript clock execution, and producer/consumer schema compatibility checks.

## Still required before release

- Production qualification of the implemented reviewer context-link commands,
  which never authorize unsupported manual values or default accounting basis.
- Broader live-service repair coverage beyond the synthetic context-link case below.
  The isolated PostgreSQL recovery test covers real transactions, runtime recreation
  and concurrent duplicate commands with controlled FinDocIQ responses. Fault tests
  additionally cover saved ingestion/artifact/workflow receipts before checkpointing.
- Broader loan policies permitting genuinely not-applicable hard-rule criteria.
  Current policies retain existing hard-rule requirements; PAT, debt/equity and
  headcount may be optional under the applicable funder's evidence policy.
- Complete source-page assessment of every uploaded financial figure, beyond the
  current read-only 805-chunk inventory and visual accounting-policy page checks.
- Full mixed-document section-boundary interpretation: unrelated statement sections
  cannot safely borrow context just because they share a PDF.
- Production PostgreSQL document-set integration/migration tests, retention of new
  set/ownership records, and deployment readiness checks for the new schema.
- Expanded real-service qualification and the full restart/outage matrix. The
  small development acceptance run below is not that production release gate.

## Deployment prerequisites

Do not enable production intake until the outstanding mandatory gates pass. Production needs
`migrations/002_document_sets.sql` applied through the reviewed migration process;
it has only been added to source, **not executed**. The local development document
root must be persistent and writable. Old unscoped ingestions are not silently
promoted into owned document sets. An explicit, verified migration/recovery path
is still needed for those existing applications.

Keep borrower data out of Git, synthetic fixtures and telemetry. Do not weaken the
basis requirement to force a private application through qualification.

## Reviewer interpretations and calculations (2026-09-11)

Evidence policy v4 permits four bounded formulas: EBITDA/revenue percentage,
CFADS/total debt service, total debt/total equity, and total debt/EBITDA. Operands
must be directly cited monetary figures with consistent entity/year/basis.
Conflicting or zero-denominator evidence cannot become a calculated answer.
The public response carries formula versions and operand receipts; FunderMatch
independently validates them. Operand metrics remain internal.

The scoped assessment interface accepts audited reviewer context links.
They identify the target chunk and declaration citations under the current
evidence policy, selecting existing evidence rather than asserting absent basis.
The masked review UI exposes bounded proposed links for reviewer confirmation.

Live qualification exposed flattened Docling OCR paragraphs. The adapter now
restores physical lines from OCR cell geometry only when the recognized token
sequence is unchanged; it cannot invent text or merge unrelated blocks.

## Measured development acceptance (2026-09-11)

One complete invented five-PDF application (six image-only pages) reached
`waiting_for_review` through live GPU OCR, Qdrant, FinDocIQ, FunderMatch and
PostgreSQL in **258.17 seconds, n=1**. All seven independently specified numeric
values matched, all four ratios carried verified calculation receipts, and absent
PAT/headcount remained null. The human decision remained unset. The wait and gold
checks survived an API process restart.

RapidOCR's instrumented smoke test observed three actual Torch inference calls on
`cuda:0`, peak allocated CUDA memory 554,728,960 bytes, and vLLM restoration. The
full intake emitted GPU release followed by `vllm_ready` about 5.14 seconds later.
FunderMatch precedent embedding in this environment uses a **CPU-only Torch wheel**;
its first retrieval stage took about 103 seconds. Cold model initialization and
downloads are included in the end-to-end time. Do not advertise this as GPU-only
latency, a production percentile, or lending/retrieval accuracy. Narrative model
generation and the full failure matrix are separate qualification gates.

Two earlier synthetic attempts stopped safely: a consumer response-model regression
and missing OCR physical lines. Both were diagnosed against actual HTTP/OCR results,
fixed, and regression-tested before the successful run. Existing borrower evidence
was not rewritten or retried.

Reproduction: generate image-only PDFs with `scripts/generate_qualification_pdfs.py`;
use FunderMatch's `scripts/qualify_synthetic_intake.py` and the independent
`scripts/verify_synthetic_qualification.py`. `--mixed-basis` generates a separate
reviewer-repair case. Keep generated files and tokens outside Git.

The separate mixed-basis case (five PDFs, seven scanned pages, **n=1**) stopped
at `needs_attention` with `evidence_basis_ambiguous` after **160.02 seconds**.
An authorized reviewer command linked the page-2 table to the explicit page-1
standalone declaration. Explicit resume reached human review in **34.33 seconds**,
with no repeated OCR. Pipeline-role access was denied and duplicate commands
returned the original result. The independent read-only audit verified exactly one
`human_reviewer` repair-request event, all seven numeric gold values, four calculation
receipts, optional absence and no lending decision. The initial audit harness used
the wrong role label; that assertion was corrected against the actual role enum and
rechecked read-only without replaying the command.
Both successful human-review waits and their gold checks survived the final API
restart. All five configured dependencies reported healthy afterward.

## Subsequent accounting extraction changes (source only)

Evidence policy `borrower-evidence-v5` adds component-based PAT, EBITDA margin,
borrowing-based leverage and explicit DSCR formula alternatives. Each calculation
requires directly cited operands with matching ownership, entity, period, basis
and units; FunderMatch independently recomputes the receipt. Cash-accrual DSCR is
available only through explicit policy selection, not a default or an automatic
lender-definition guess. EBITDA is not EBITA or adjusted operating EBITDA.

Unresolved assessments expose candidate citations and per-operand safe diagnostic
codes. The UI describes failed extraction as not verified, not proof of absence.
Alphabetic table row markers and a Udyam/Aadhaar classification collision are
covered by regressions. An invented two-page PDF exercises actual parsing,
chunking, cross-page explicit basis and component calculations.

Verification: FinDocIQ 205 tests passed; FunderMatch 182 passed, 2 skipped;
Ruff, generated-contract freshness, consumer compatibility and JavaScript syntax
checks passed. These totals are not live-service qualification.

These changes have not been deployed or used to retry existing applications.
The earlier live measurements above qualify the preceding implementation only.
The subsequent source revision adds an audited reporting-scope authority through
existing reviewer repair commands; see `basis-resolution-design.md`. It is not
an automatic accounting-basis override. It also reconciles reported annual
debt-service totals and permits an explicitly selected cash-accrual/PBT formula.
The September 16 clarification changes implement dated incorporation age,
preserved multi-activity NIC extraction, explicitly scoped third-party financial
sources, image OCR and annual debt-service reconciliation. Primary NIC selection
is intentionally not invented. See `borrower-policy-clarifications.md` for the
current implementation and acceptance limits; the earlier results above describe
the preceding revision only.
