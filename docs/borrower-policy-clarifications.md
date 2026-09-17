# Borrower intake policy clarifications

## Implemented source contracts

- PDF, JPEG and PNG intake. Images use a bounded, temporary single-page PDF OCR
  adapter; identity and citations retain the original image hash. Image boxes use
  the wrapper's PDF-point coordinates and page dimensions, not raw pixel offsets.
  Original files are not rewritten. Multi-frame, malformed and oversized images
  are rejected before model execution. XLS/XLSX are not accepted by this interface.
- Company age is completed incorporation years at a persisted assessment date.
  It requires explicitly labelled company PAN or registration evidence. It is not
  commencement of operations or a personal date of birth. The consumer recomputes
  the dated receipt and checks the requested assessment date.
- Udyam NIC tables retain all 2/4/5-digit codes, their descriptions and activities.
  Hierarchy conflicts stop extraction. No primary activity is invented. Mapping
  version `udyam-nic-hierarchy-v1` denotes hierarchy/description selection, not a
  separately certified government industry taxonomy.
- An approved third-party report remains `financial_information_report`, never
  `audited_financials`. `FINDOCIQ_SOURCE_APPROVALS_FILE` names an operator-controlled
  YAML file outside Git containing application ID, original document SHA-256,
  provider `probe42`, approval ID, approving actor and reason. The owner explicitly
  authorizes each scope. Removing an approval changes the evidence policy hash.
  This records source acceptance, not independent authentication of the provider.
- The approved catalogue additionally includes profit on sale of fixed assets,
  other/interest income, current assets/liabilities, inventories, receivables,
  payables, cash equivalents, operating cash flow and financial formula operands.
  Blank cells are not zero. Each result requires its own source/context receipt.

## DSCR

DSCR is calculation-only. The default numerator is explicitly supported CFADS;
cash-accrual alternatives require deliberate policy selection. EBITDA, EBIT and
adjusted CFADS are not interchangeable automatic fallbacks.

`dscr_cfads_all_obligations_v1` divides CFADS by annual principal due on all loans,
interest due on all loans, and mandatory lease payments explicitly excluded from
those loan obligations. Every component must be cited, including an explicit zero
where applicable. A partially disclosed all-obligation schedule cannot fall back
to a narrower term-debt denominator. Closing debt balances and cash-flow financing
outflows are not substitutes for annual obligations due.

These are historical-period calculations. A proposed-facility DSCR is not inferred
from a requested amount: pricing, utilization, amortization and relevant lease
commitments are required. Do not describe a historical DSCR as including a new loan.

## Workflow and qualification

FunderMatch now distinguishes secured/unsecured products. Missing collateral fails
secured-product checks; it is not required for an unsecured product. Employee count
is optional unless an applicable funder policy requires it. Evidence review exposes
optional/not-applicable versus unresolved required fields, source citations, formula
and operand citations, NIC details and approved-provider identity.

Cross-page accounting-basis resolution remains conservative. Contradictory or
unlinked scope requires the existing audited reviewer-repair operation; neither an
LLP suffix nor absence of a consolidated heading establishes standalone basis.

Automated gates at this revision: FinDocIQ 244 passed; FunderMatch 192 passed,
2 skipped; lint, public-contract freshness/consumer compatibility and browser
syntax checks passed. These counts do not establish live OCR or borrower-case
acceptance. Private source PDFs/images and gold values remain outside Git and evals.

Real-source verification also exposed and corrected two report-layout defects:
explicit `Standalone Financial Data` headings with dated statement columns, and
older annexures being validated before latest-period selection. A header-only
business-activity table now recovers body cells using its actual column/rule
geometry and the next section boundary. A separate regression covers that PDF
layout. No source values or eval expectations were changed.
