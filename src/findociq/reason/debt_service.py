"""Reconcile explicitly annual debt-service disclosures; never infer from balances."""

from decimal import Decimal, localcontext


def reconcile_annual_service(extract):
    from findociq.reason.evidence_gate import EvidenceInsufficient

    complete = []
    for name in (
        "all_loan_principal_due_crore",
        "all_loan_interest_due_crore",
        "additional_lease_service_crore",
    ):
        try:
            complete.append(extract(name))
        except EvidenceInsufficient as error:
            if error.code != "evidence_metric_missing":
                raise
    if complete:
        # Never ignore a partially disclosed broader obligation schedule and
        # fall back to a narrower term-debt denominator. Absence is not zero.
        if len(complete) != 3:
            raise EvidenceInsufficient("evidence_metric_missing")
        receipts = [v.evidence_validation for v in complete]
        if (
            any(r is None for r in receipts)
            or len({(r.entity_hash, r.period, r.basis, r.policy_hash) for r in receipts}) != 1
        ):
            raise EvidenceInsufficient("evidence_context_conflict")
        amounts = [Decimal(v.value) for v in complete]
        if any(v < 0 for v in amounts) or sum(amounts) <= 0:
            raise EvidenceInsufficient("evidence_value_ambiguous")
        try:
            total = extract("debt_service_crore")
        except EvidenceInsufficient as error:
            if error.code != "evidence_metric_missing":
                raise
        else:
            r = total.evidence_validation
            if r is None or (r.entity_hash, r.period, r.basis, r.policy_hash) != (
                receipts[0].entity_hash,
                receipts[0].period,
                receipts[0].basis,
                receipts[0].policy_hash,
            ):
                raise EvidenceInsufficient("evidence_context_conflict")
            if Decimal(total.value) != sum(amounts):
                raise EvidenceInsufficient("evidence_conflicting_values")
        return
    values = []
    missing = False
    for name in ("debt_service_crore", "principal_due_crore", "term_interest_due_crore"):
        try:
            values.append(extract(name))
        except EvidenceInsufficient as error:
            if error.code != "evidence_metric_missing":
                raise
            missing = True
    # No partial sum, zero-fill, annualisation, or cash-flow proxy is permitted.
    # The selected DSCR formula will separately require all its own operands.
    if missing:
        return
    receipts = [v.evidence_validation for v in values]
    if (
        any(r is None for r in receipts)
        or len({(r.entity_hash, r.period, r.basis, r.policy_hash) for r in receipts}) != 1
    ):
        raise EvidenceInsufficient("evidence_context_conflict")
    with localcontext() as context:
        context.prec = 28
        total, principal, interest = (Decimal(v.value) for v in values)
        if total <= 0 or principal < 0 or interest < 0 or total != principal + interest:
            raise EvidenceInsufficient("evidence_conflicting_values")
