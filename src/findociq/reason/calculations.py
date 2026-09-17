"""Versioned formulas; no evaluation of arbitrary expressions or inferred operands."""

from collections.abc import Callable
from decimal import Decimal

from findociq.reason.accounting_formulas import FORMULAS
from findociq.reason.schema import CalculationReceipt, ExtractedFigure


def calculate(
    formula_id: str, extract: Callable[[str], ExtractedFigure], policy_hash: str
) -> ExtractedFigure:
    from findociq.reason.evidence_gate import EvidenceInsufficient

    if formula_id not in FORMULAS:
        raise EvidenceInsufficient("metric_not_allowed")
    formula = FORMULAS[formula_id]
    metric, unit = formula.metric, formula.unit
    operands = tuple(extract(name) for name in formula.operands)
    receipts = [f.evidence_validation for f in operands]
    if (
        any(
            r is None
            or r.policy_hash != policy_hash
            or f.calculation is not None
            or f.unit != "INR crore"
            or not (
                r.document_type == "audited_financials"
                or (
                    r.document_type == "financial_information_report"
                    and r.source_provider == "probe42"
                    and r.source_approval_id
                )
            )
            or r.basis not in {"standalone", "consolidated"}
            or not r.period
            for f, r in zip(operands, receipts, strict=True)
        )
        or len({(r.entity_hash, r.period, r.basis) for r in receipts}) != 1
    ):
        raise EvidenceInsufficient("evidence_context_conflict")
    numbers = tuple(Decimal(f.value) for f in operands)
    try:
        value = format(formula.evaluate(numbers), "f")
    except ValueError as error:
        raise EvidenceInsufficient("evidence_value_ambiguous") from error
    receipt = receipts[0].model_copy(
        update={
            "metric_id": metric,
            "source_value": "",
            "source_unit": "calculation",
            "conversion_factor": "1",
        }
    )
    return ExtractedFigure(
        label=metric,
        value=value,
        unit=unit,
        period=operands[0].period,
        citation=operands[0].citation,
        evidence_validation=receipt,
        calculation=CalculationReceipt(
            formula_id=formula_id,
            policy_hash=policy_hash,
            operands=operands,
        ),
    )
