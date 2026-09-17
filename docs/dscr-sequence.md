# DSCR evidence sequence

The default borrower evidence policy uses NOI, not a mandatory printed DSCR or
CFADS value. NOI is defined here as operating revenue less operating expenses
including depreciation, excluding finance costs and tax. It is not EBITDA.
This is a versioned policy definition, not a universal lender formula.

1. Assess cited numerator components first. Prefer explicitly scoped operating
   expenses; alternatively use operating revenue minus pre-tax P&L total
   expenses plus finance costs. Total income cannot replace operating revenue.
   A supported explicitly reported NOI is another configured route.
2. Assess same-period total debt service or the full principal, interest and
   additional lease schedule. Closing debt balances are not repayments.
   Missing obligations, including lease amounts, are never assumed zero.
3. Require matching entity, reporting period and accounting basis for operands.
   Conflicting supported routes require evidence review.
4. Calculate deterministically; retain the formula ID, policy hash and cited
   operands. FunderMatch independently verifies the receipt and arithmetic.
5. Return numerator/debt-service operand diagnostics for unresolved cases.
   A missing extracted operand means it was not verified, not that the source
   document lacks it. Complete the evidence-repair process before reassessment.

CFADS and cash-accrual formulas remain explicitly selectable policies. They
cannot be mixed with NOI fallback routes. Newly proposed debt also requires
documented terms; historical accounts alone cannot establish prospective
debt service.

Qualification covers deterministic extraction and receipt tests. It does not
prove every real accounting layout is supported. Exceptional/non-operating
expense adjustments need explicit policy and supporting evidence; the simple
P&L route must not be presented as a full lender-specific adjusted-cash-flow
assessment. Deploy both matching policy bundles together after release checks.
