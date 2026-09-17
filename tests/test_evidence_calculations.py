from decimal import Decimal, localcontext

import pytest
from test_evidence_gate import gate, statement  # noqa: F401

from findociq.reason.evidence_gate import EvidenceInsufficient


def test_margin_uses_explicit_cited_operands(gate):  # noqa: F811
    source = statement(row="| Revenue from operations | 1200 | 900 |\n| EBITDA | 120 | 90 |")
    result = gate.extract("ebitda_margin_pct", (source,))
    assert Decimal(result.value) == 10
    assert result.calculation.formula_id == "ebitda_margin_v1"
    assert [o.value for o in result.calculation.operands] == ["1.20", "12.00"]
    assert all(o.citation == result.citation for o in result.calculation.operands)
    with localcontext() as ctx:
        ctx.prec = 3
        assert gate.extract("ebitda_margin_pct", (source,)) == result


def test_missing_dscr_never_uses_profit_as_proxy(gate):  # noqa: F811
    source = statement(row="| Profit after tax | 120 | 90 |\n| Total debt service | 50 | 40 |")
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract("dscr", (source,))


def test_borrowing_leverage_uses_explicit_operating_ebitda(gate):  # noqa: F811
    source = statement(
        row=(
            "| Long term borrowings | 300 | 200 |\n"
            "| Short term borrowings | 100 | 80 |\n"
            "| Operating Profit ( EBITDA ) | 200 | 140 |"
        )
    )
    result = gate.extract("debt_to_ebitda", (source,))
    assert Decimal(result.value) == 2
    assert result.calculation.formula_id == "debt_ebitda_borrowings_v1"


@pytest.mark.parametrize("denominator", ["0", "-1"])
def test_invalid_denominator_abstains(gate, denominator):  # noqa: F811
    source = statement(
        row=(
            "| Cash flow available for debt service | 100 | 90 |\n"
            f"| Total debt service | {denominator} | 40 |"
        )
    )
    with pytest.raises(EvidenceInsufficient, match="value_ambiguous"):
        gate.extract("dscr", (source,))


def test_direct_conflicting_ratio_cannot_be_bypassed(gate):  # noqa: F811
    source = statement(
        row=(
            "| DSCR | 2 | 1 |\n| DSCR | 3 | 1 |\n"
            "| Cash flow available for debt service | 100 | 90 |\n"
            "| Total debt service | 50 | 40 |"
        )
    )
    with pytest.raises(EvidenceInsufficient, match="conflicting_values"):
        gate.extract("dscr", (source,))
