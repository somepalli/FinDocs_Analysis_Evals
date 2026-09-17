from decimal import Decimal

import pytest
from test_evidence_gate import gate, statement  # noqa: F401

from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, EvidencePolicy


def components():
    return (
        "| Revenue from operations | 1000 | 900 |\n"
        "| Total expenses | 850 | 780 |\n"
        "| Finance costs | 50 | 40 |\n"
        "| Profit before tax | 180 | 150 |"
    )


def test_default_derives_noi_from_pnl_not_cfads_or_reported_dscr(gate):  # noqa: F811
    result = gate.extract(
        "dscr", (statement(row=components() + "\n| Total debt service | 100 | 90 |"),)
    )
    assert Decimal(result.value) == 2
    assert result.calculation.formula_id == "dscr_noi_pnl_components_total_v1"
    assert [o.label for o in result.calculation.operands] == [
        "operating_revenue_crore",
        "pnl_total_expenses_crore",
        "finance_cost_crore",
        "debt_service_crore",
    ]
    assert all(o.citation.document_id == "financials" for o in result.calculation.operands)


def test_sequence_reads_all_numerator_routes_before_debt_service(gate, monkeypatch):  # noqa: F811
    calls = []
    direct = gate._direct

    def tracked(name, chunks):
        calls.append(name)
        return direct(name, chunks)

    monkeypatch.setattr(gate, "_direct", tracked)
    gate.extract("dscr", (statement(row=components() + "\n| Total debt service | 100 | 90 |"),))
    assert calls.index("finance_cost_crore") < calls.index("debt_service_crore")
    assert calls.count("operating_revenue_crore") == 1


def test_missing_schedule_reports_exact_operands_and_both_stages(gate):  # noqa: F811
    result = gate.assess(("dscr",), (statement(row=components()),))[0]
    assert result.status == "missing"
    route = [
        d for d in result.operand_diagnostics if d.formula_id == "dscr_noi_pnl_components_total_v1"
    ]
    assert all(d.error_code is None for d in route if d.stage == "numerator")
    assert [(d.metric_id, d.error_code) for d in route if d.stage == "debt_service"] == [
        ("debt_service_crore", "evidence_metric_missing")
    ]
    assert not any(d.metric_id == "cfads_crore" for d in result.operand_diagnostics)


def test_full_schedule_includes_explicit_additional_leases(gate):  # noqa: F811
    rows = components() + (
        "\n| Total scheduled principal due on all loans for the year | 60 | 50 |"
        "\n| Total interest due on all loans for the year | 30 | 30 |"
        "\n| Mandatory lease payments not included in loan debt service for the year | 10 | 10 |"
    )
    result = gate.extract("dscr", (statement(row=rows),))
    assert Decimal(result.value) == 2
    assert len(result.calculation.operands) == 6
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract("dscr", (statement(row=rows.rsplit("\n", 1)[0]),))


def test_no_ebitda_or_cfads_substitution_under_noi_policy(gate):  # noqa: F811
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract(
            "dscr",
            (
                statement(
                    row=(
                        "| EBITDA | 200 | 150 |\n"
                        "| Cash flow available for debt service | 200 | 150 |\n"
                        "| Total debt service | 100 | 90 |"
                    )
                ),
            ),
        )


def test_pnl_expense_component_requires_pbt_context(gate):  # noqa: F811
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract(
            "dscr",
            (
                statement(
                    row=(
                        "| Revenue from operations | 1000 | 900 |\n"
                        "| Total expenses | 850 | 780 |\n| Finance costs | 50 | 40 |\n"
                        "| Total debt service | 100 | 90 |"
                    )
                ),
            ),
        )


def test_noi_and_cfads_definitions_cannot_mix(gate):  # noqa: F811
    config = gate.policy.model_dump()
    config["calculation_alternatives"]["dscr"] = ["dscr_cfads_v1"]
    with pytest.raises(ValueError, match="definitions cannot be mixed"):
        BorrowerEvidenceGate(EvidencePolicy.model_validate(config))


@pytest.mark.parametrize("debt", ["0", "-10"])
def test_nonpositive_debt_service_is_not_a_ratio(gate, debt):  # noqa: F811
    with pytest.raises(EvidenceInsufficient, match="value_ambiguous"):
        gate.extract(
            "dscr", (statement(row=components() + f"\n| Total debt service | {debt} | 90 |"),)
        )


def test_total_income_cannot_replace_operating_revenue(gate):  # noqa: F811
    rows = components().replace("Revenue from operations", "Total revenue")
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract("dscr", (statement(row=rows + "\n| Total debt service | 100 | 90 |"),))


def test_conflicting_noi_routes_require_review(gate):  # noqa: F811
    rows = (
        components() + "\n| Net operating income | 250 | 150 |\n| Total debt service | 100 | 90 |"
    )
    with pytest.raises(EvidenceInsufficient, match="context_conflict"):
        gate.extract("dscr", (statement(row=rows),))
