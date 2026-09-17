"""Staged DSCR assessment over cited operands, never a copied reported ratio."""

from decimal import Decimal

from findociq.reason.accounting_formulas import FORMULAS
from findociq.reason.calculations import calculate
from findociq.reason.debt_service import reconcile_annual_service


def calculate_dscr(gate, chunks):
    from findociq.reason.evidence_gate import EvidenceInsufficient

    primary = gate.policy.calculations.get("dscr")
    if primary is None:
        raise EvidenceInsufficient("metric_not_allowed")
    choices = (primary, *gate.policy.calculation_alternatives.get("dscr", ()))
    cache = {}

    def extract(name):
        if name not in cache:
            try:
                cache[name] = gate._direct(name, chunks)
            except EvidenceInsufficient as error:
                cache[name] = error
        result = cache[name]
        if isinstance(result, EvidenceInsufficient):
            raise result
        return result

    # All numerator routes are assessed first. Missing one representation never
    # stops assessment of the other supported component representations.
    # Debt-service discovery follows even when the numerator is unresolved.
    for side in ("numerator", "denominator"):
        for formula_id in choices:
            formula = FORMULAS[formula_id]
            for name, coefficient in zip(formula.operands, getattr(formula, side), strict=True):
                if coefficient:
                    try:
                        extract(name)
                    except EvidenceInsufficient as error:
                        if error.code == "evidence_scope_violation":
                            raise

    # Ambiguous/conflicting evidence cannot be bypassed using another formula.
    for result in cache.values():
        if isinstance(result, EvidenceInsufficient) and result.code != "evidence_metric_missing":
            raise result
    reconcile_annual_service(extract)
    for formula_id in choices:
        formula = FORMULAS[formula_id]
        names = [
            n
            for n, coefficient in zip(formula.operands, formula.denominator, strict=True)
            if coefficient
        ]
        found = [cache[n] for n in names if not isinstance(cache[n], EvidenceInsufficient)]
        if any(Decimal(f.value) < 0 for f in found) or (
            len(found) == len(names) and sum(Decimal(f.value) for f in found) <= 0
        ):
            raise EvidenceInsufficient("evidence_value_ambiguous")
    # A ratio is not required or returned, but conflicting disclosures still stop.
    try:
        extract("dscr")
    except EvidenceInsufficient as error:
        if error.code != "evidence_metric_missing":
            raise
    results = []
    for formula_id in choices:
        try:
            results.append(calculate(formula_id, extract, gate.policy.policy_hash))
        except EvidenceInsufficient as error:
            if error.code != "evidence_metric_missing":
                raise
    if results:
        if len({Decimal(result.value) for result in results}) != 1:
            raise EvidenceInsufficient("evidence_context_conflict")
        return results[0]
    raise EvidenceInsufficient("evidence_metric_missing")
