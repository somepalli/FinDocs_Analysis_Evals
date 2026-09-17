from decimal import Decimal

import pytest
from test_evidence_gate import gate, statement  # noqa: F401

from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient


def test_realistic_pat_label_and_pbt_tax_fallback(gate):  # noqa: F811
    explicit = statement(row="| 10 | Profit/(Loss) for the year after tax (8-9) | 75 | 60 |")
    # Include the serial-number column in the header too.
    explicit = explicit.model_copy(
        update={"text": explicit.text.replace("| Particulars |", "| | Particulars |")}
    )
    assert Decimal(gate.extract("pat_crore", (explicit,)).value) == Decimal("0.75")
    source = statement(
        row=(
            "| Profit/(Loss) before tax (6 - 7) | 100 | 80 |\n"
            "| Net tax expense / (benefit) | 25 | 20 |"
        )
    )
    result = gate.extract("pat_crore", (source,))
    assert Decimal(result.value) == Decimal("0.75")
    assert result.calculation.formula_id == "pat_pbt_net_tax_v1"


def test_margin_from_accounting_components_keeps_all_leaf_receipts(gate):  # noqa: F811
    source = statement(
        row=(
            "| Revenue from operations | 1000 | 800 |\n"
            "| Profit/(Loss) before tax | 100 | 80 |\n| Finance costs | 20 | 15 |\n"
            "| Depreciation and amortisation expense | 10 | 8 |"
        )
    )
    result = gate.extract("ebitda_margin_pct", (source,))
    assert Decimal(result.value) == 13
    assert len(result.calculation.operands) == 4
    assert all(item.calculation is None for item in result.calculation.operands)


def test_llp_debt_equity_from_balance_sheet_components(gate):  # noqa: F811
    source = statement(
        row=(
            "| Long term borrowings | 200 | 180 |\n| Short term borrowings | 100 | 90 |\n"
            "| Partners' Contribution | 100 | 90 |\n| Partners' Current Account | 200 | 180 |"
        )
    )
    result = gate.extract("debt_to_equity", (source,))
    assert Decimal(result.value) == 1
    assert len(result.calculation.operands) == 4


def test_cash_accrual_dscr_requires_explicit_policy_and_all_schedule_operands(gate):  # noqa: F811
    source = statement(
        row=(
            "| Profit after tax | 75 | 60 |\n| Depreciation and amortisation expense | 10 | 8 |\n"
            "| Interest due on term debt | 15 | 12 |\n"
            "| Scheduled principal repayments due | 35 | 30 |"
        )
    )
    with pytest.raises(EvidenceInsufficient):
        gate.extract("dscr", (source,))
    policy = gate.policy.model_copy(
        update={
            "calculations": {**gate.policy.calculations, "dscr": "dscr_cash_accrual_v1"},
            "calculation_alternatives": {},
        }
    )
    result = BorrowerEvidenceGate(policy).extract("dscr", (source,))
    assert Decimal(result.value) == 2


def test_operand_diagnostics_never_call_not_verified_absent(gate):  # noqa: F811
    source = statement(row="| Profit before tax | 100 | 80 |\n| Finance costs | 20 | 15 |")
    field = gate.assess(("ebitda_margin_pct",), (source,))[0]
    inputs = {d.metric_id: d.error_code for d in field.operand_diagnostics}
    assert inputs["pbt_crore"] is None
    assert inputs["da_crore"] == "evidence_metric_missing"
    assert field.figure is None


def test_calculation_does_not_bypass_conflicting_component(gate):  # noqa: F811
    source = statement(
        row=(
            "| Profit before tax | 100 | 80 |\n| Net tax expense | 25 | 20 |\n"
            "| Net tax expense | 30 | 20 |"
        )
    )
    with pytest.raises(EvidenceInsufficient, match="conflicting_values"):
        gate.extract("pat_crore", (source,))
