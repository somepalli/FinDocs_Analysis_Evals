from decimal import Decimal

import pytest
from test_evidence_gate import gate, statement  # noqa: F401


@pytest.mark.parametrize(
    "metric,label",
    [
        ("profit_on_sale_fixed_assets_crore", "Profit on sale / disposal of Fixed Assets"),
        ("current_assets_crore", "Total current assets"),
        ("operating_cash_flow_crore", "Net cash from operating activities"),
    ],
)
def test_catalogue_facts_keep_value_year_units_and_citation(gate, metric, label):  # noqa: F811
    source = statement(units="Amount in Rs.", row=f"| {label} | 2705 | |")
    result = gate.extract(metric, (source,))
    assert Decimal(result.value) == Decimal("0.0002705")
    assert result.period == "2025"
    assert result.citation.document_id == "financials"
    assert result.evidence_validation.chunk_id == source.chunk_id
    assert result.evidence_validation.source_value == "2705"


def test_catalogue_does_not_zero_fill_blank_latest_value(gate):  # noqa: F811
    source = statement(row="| Profit on sale of fixed assets | | 50 |")
    field = gate.assess(("profit_on_sale_fixed_assets_crore",), (source,))[0]
    assert field.status != "supported"
    assert field.figure is None
