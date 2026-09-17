from decimal import Decimal
from pathlib import Path

import pytest
from test_evidence_gate import statement

from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, EvidencePolicy


@pytest.fixture
def gate():
    """The original CFADS suite explicitly selects the retained legacy policy."""
    config = EvidencePolicy.load(Path("configs/evidence/borrower.yaml")).model_dump()
    config["calculations"]["dscr"] = "dscr_cfads_v1"
    config["calculation_alternatives"]["dscr"] = [
        "dscr_cfads_all_obligations_v1",
        "dscr_cfads_schedule_v1",
    ]
    return BorrowerEvidenceGate(EvidencePolicy.model_validate(config))


def test_reported_dscr_without_operands_is_not_accepted(gate):  # noqa: F811
    source = statement(row="| DSCR | 2.00 | 1.80 |")
    with pytest.raises(EvidenceInsufficient, match="evidence_metric_missing"):
        gate.extract("dscr", (source,))
    assessment = gate.assess(("dscr",), (source,))[0]
    assert assessment.status == "missing"
    assert assessment.operand_diagnostics


def test_dscr_includes_explicit_additional_lease_obligations(gate):  # noqa: F811
    rows = (
        "| Cash flow available for debt service | 100 | 80 |\n"
        "| Total scheduled principal due on all loans for the year | 30 | 24 |\n"
        "| Total interest due on all loans for the year | 10 | 8 |\n"
        "| Mandatory lease payments not included in loan debt service for the year | 10 | 8 |"
    )
    result = gate.extract("dscr", (statement(row=rows),))
    assert Decimal(result.value) == 2
    assert result.calculation.formula_id == "dscr_cfads_all_obligations_v1"
    assert len(result.calculation.operands) == 4
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract("dscr", (statement(row=rows.rsplit("\n", 1)[0]),))


def test_dscr_is_computed_with_cited_operands_even_when_ratio_is_reported(gate):  # noqa: F811
    rows = (
        "| DSCR | 2.00 | 2.00 |\n"
        "| Cash flow available for debt service | 100 | 80 |\n"
        "| Total debt service | 50 | 40 |"
    )
    result = gate.extract("dscr", (statement(row=rows),))
    assert Decimal(result.value) == 2
    assert result.calculation is not None
    assert result.calculation.formula_id == "dscr_cfads_v1"
    assert len(result.calculation.operands) == 2
    assert all(o.citation.document_id == "financials" for o in result.calculation.operands)


def test_dscr_does_not_fall_back_to_reported_ratio_without_formula(gate):  # noqa: F811
    config = gate.policy.model_dump()
    config["calculations"].pop("dscr")
    config["calculation_alternatives"].pop("dscr")
    selected = BorrowerEvidenceGate(EvidencePolicy.model_validate(config))
    with pytest.raises(EvidenceInsufficient, match="metric_not_allowed"):
        selected.extract("dscr", (statement(row="| DSCR | 2 | 2 |"),))


def test_annual_service_total_must_match_components(gate):  # noqa: F811
    rows = (
        "| Cash flow available for debt service | 100 | 80 |\n"
        "| Total debt service | 50 | 40 |\n"
        "| Scheduled principal repayments due | 35 | 30 |\n"
        "| Interest due on term debt | 15 | 10 |"
    )
    assert Decimal(gate.extract("dscr", (statement(row=rows),)).value) == 2
    with pytest.raises(EvidenceInsufficient, match="conflicting_values"):
        gate.extract("dscr", (statement(row=rows.replace("| 50 |", "| 60 |")),))


def test_closing_maturities_and_cashflow_repayments_are_not_annual_due(gate):  # noqa: F811
    rows = (
        "| Cash flow available for debt service | 100 | 80 |\n"
        "| Current maturities of long term debt | 35 | 30 |\n"
        "| Repayment of long term borrowings | 35 | 30 |\n"
        "| Finance costs | 15 | 10 |"
    )
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract("dscr", (statement(row=rows),))


def test_selected_cash_accrual_definition_can_calculate_pat_from_components(gate):  # noqa: F811
    config = gate.policy.model_dump()
    config["calculations"]["dscr"] = "dscr_cash_accrual_v1"
    config["calculation_alternatives"]["dscr"] = ["dscr_cash_accrual_pbt_v1"]
    selected = BorrowerEvidenceGate(EvidencePolicy.model_validate(config))
    rows = (
        "| Profit before tax | 100 | 80 |\n| Net tax expense | 25 | 20 |\n"
        "| Depreciation and amortisation expense | 10 | 8 |\n"
        "| Interest due on term debt | 15 | 12 |\n"
        "| Scheduled principal repayments due | 35 | 30 |"
    )
    assert Decimal(selected.extract("dscr", (statement(row=rows),)).value) == 2
    config["calculation_alternatives"]["dscr"] = ["dscr_cfads_v1"]
    with pytest.raises(ValueError, match="definitions cannot be mixed"):
        EvidencePolicy.model_validate(config)
