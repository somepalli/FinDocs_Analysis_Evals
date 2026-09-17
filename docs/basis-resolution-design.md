# Safe accounting-basis resolution for real borrower reports

Status: reporting-scope determination implemented in source; not deployed or
live-release-qualified. Existing explicit-declaration checks remain.

## Separate observation from acceptance

Retain candidate row/page/bbox and per-operand diagnostics outside checkpoints even
when accounting context is unresolved. A failed extraction means **not verified**,
not that the source document lacks the fact. No unresolved candidate may enter
eligibility or a calculation receipt marked accepted.

## Resolution order

1. Explicit table/statement heading.
2. Cited enclosing statement section with matching entity and reporting period.
3. Explicit linked declaration within the same report.
4. If the report does not state standalone/consolidated, require an audited human
   reporting-scope determination, not a boolean override or an invented declaration.

The fourth route is a **distinct authority type**, separate from the existing
`reviewer_context_link` that only links an explicit declaration. Its evidence must
include the audit opinion, entity-named balance sheet/P&L, basis-of-preparation note,
report period and page ranges. A reviewer confirms that these comprise the named
entity's financial statements. Absence of the word consolidated, LLP legal form,
or a bookkeeping clause in an agreement is never sufficient on its own.

If group scope, subsidiary coverage, entity identity or period is contradictory,
remain at needs_attention and request auditor/borrower clarification. Do not let
the new authority overwrite an explicit contradictory declaration.

## Command and receipt

Reviewer-only command: application ID, immutable document-set/hash, document/section
IDs, entity hash, period, selected scope, cited audit/title/policy pages, reason,
expected workflow version, stable command ID and policy version. Store in the
authoritative PostgreSQL audit, never as a model-generated decision. The receipt
must distinguish **reviewer determination** from **source explicitly states basis**.
FinDocIQ verifies ownership, geometry, period and evidence linkage; FunderMatch
verifies the authorized command and receipt. Checkpoints carry only receipt IDs.

Document additions, relevant page changes, period changes and policy changes
invalidate the determination and downstream calculations. Reuse completed OCR;
reassess all affected fields, eligibility and suggestions. No automatic lending
decision is permitted.

Implemented through the existing versioned evidence-interpretations command:
`authority=reporting_scope` plus the entity hash and target/context citations.
The receipt uses `reviewer_reporting_scope`, not `reviewer_context_link`.
Only entity-only scope is offered; consolidated/mixed scope requires explicit
declarations. Audit-scope text, audited period, report owner, statement anchor and
preparation note must resolve. A related-party relationship is not itself a
consolidation declaration, and the predecessor-audit year is not the current
opinion period. Choices are never selected or submitted automatically.

Read-only acceptance against the private application's nine documents / 1,051
chunks produced four unapproved scope choices. No human decision, application
retry, service restart or borrower-file modification was performed.

## Accounting policy must also be explicit

- PAT: reported PAT, otherwise cited PBT minus **net** tax expense, including a
  documented signed tax benefit. Current tax alone is not automatically net tax.
- EBITDA: selected policy is PBT + finance cost + depreciation/amortisation. This
  can include other income; it must not be labelled adjusted operating EBITDA.
  EBITA is different and is not substituted for EBITDA.
- Borrowings-based debt/equity: long-term plus short-term borrowings, divided by
  reported equity or explicitly classified LLP contribution/current-account equity.
  Current maturities included in short-term borrowings must not be added again.
  This is not a lease-adjusted or all-liabilities definition of debt.
- DSCR: select the lender's definition. CFADS/debt service is the default; a
  cash-accrual formula is available only by explicit policy selection and needs
  PAT, D&A, term-debt interest and scheduled principal for the same period. Closing
  current maturities and net loan cashflows are not automatically annual principal
  due. Missing operands remain not verified; a partial formula is not a result.

Annual reported total debt service is checked against independently extracted
annual principal due plus term interest when all three disclosures are available;
conflicts stop extraction. This does not reconstruct arbitrary monthly bank
schedules or infer scheduled amounts from cash-flow repayments. Those layouts
remain unresolved pending a supported schedule contract and acceptance fixtures.
To select cash-accrual DSCR explicitly, set `calculations.dscr` to
`dscr_cash_accrual_v1` and its alternatives to `[dscr_cash_accrual_pbt_v1]` in the
versioned evidence policy. Policy validation rejects mixing CFADS and cash-accrual
definitions as fallbacks. This is a deployment policy choice, not automatic
per-funder routing or a reviewer-entered financial value.

## Release tests required for the new authority

Entity-only audit without explicit basis; consolidated report; mixed sections;
two entities; amended/restated report; same-year conflicts; unrelated agreement;
missing audit note; stale/replayed command; cross-application citations; API restart;
policy change and new-document invalidation. Only an authorized, unambiguous,
fully cited determination may release basis-blocked candidates.
