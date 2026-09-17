from pathlib import Path

import pytest

from findociq.ingest.schema import BoundingBox, Provenance, TextChunk
from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, EvidencePolicy


def page(text, number=1, document="report"):
    return TextChunk(
        chunk_id=f"{document}-{number}",
        text=text,
        provenance=(
            Provenance(
                document_id=document,
                source_path="synthetic.pdf",
                page_number=number,
                page_width=600,
                page_height=800,
                bbox=BoundingBox(x0=10, y0=10, x1=550, y1=700),
            ),
        ),
    )


def sources(declaration="Standalone financial statements\nYear ended FY2024-25"):
    return (
        page("Audited financial statements\nLegal name: Example Manufacturing LLP\n" + declaration),
        page(
            "Amounts in lakhs\n| Particulars | FY2024-25 | FY2023-24 |\n"
            "| Revenue from operations | 1200 | 900 |",
            2,
        ),
    )


def gate():
    return BorrowerEvidenceGate(EvidencePolicy.load(Path("configs/evidence/borrower.yaml")))


def test_report_heading_carries_separate_context_provenance():
    result = gate().extract("annual_revenue_crore", sources())
    assert result.value == "12.00"
    assert result.citation.page_number == 2
    assert result.evidence_validation.basis_resolution == "report_declaration"
    assert result.evidence_validation.context_citations[0].page_number == 1


def test_explicit_financial_data_heading_links_period_context():
    result = gate().extract(
        "annual_revenue_crore", sources("Standalone Financial Data\nYear ended FY2024-25")
    )
    assert result.evidence_validation.basis == "standalone"
    assert result.evidence_validation.context_citations[0].page_number == 1


def test_financial_data_heading_uses_dated_statement_columns_not_maturity_dates():
    from findociq.reason.evidence_gate import _report_years

    heading = "| Balance Sheet - AOC-4 (Rs. Crore) | 31 Mar, 2024 | 31 Mar, 2025 |"
    assert _report_years(heading) == (2024, 2025)
    assert _report_years("| Loan maturity | 31 Mar, 2027 |") == ()
    result = gate().extract(
        "annual_revenue_crore", sources("Standalone Financial Data\n" + heading)
    )
    assert result.evidence_validation.basis == "standalone"


@pytest.mark.parametrize(
    "declaration",
    [
        "Standalone financial statements\nYear ended FY2021-22",
        "This agreement requires consolidated accounts.\nYear ended FY2024-25",
        "Standalone financial statements\nConsolidated financial statements\nYear ended FY2024-25",
        "Year ended FY2024-25",
    ],
)
def test_unlinked_or_conflicting_context_does_not_fill_basis(declaration):
    with pytest.raises(EvidenceInsufficient, match="basis_ambiguous"):
        gate().extract("annual_revenue_crore", sources(declaration))


def test_another_document_cannot_supply_basis():
    chunks = sources("Year ended FY2024-25") + (
        page(
            "LLP agreement\nStandalone financial statements\nYear ended FY2024-25", document="proof"
        ),
    )
    with pytest.raises(EvidenceInsufficient, match="basis_ambiguous"):
        gate().extract("annual_revenue_crore", chunks)


def test_old_annexure_without_linked_basis_cannot_veto_current_year():
    old = page(
        "Amounts in lakhs\n| Particulars | FY2018-19 | FY2017-18 |\n"
        "| Revenue from operations | 500 | 400 |",
        30,
    )
    current = sources()
    result = gate().extract("annual_revenue_crore", (old, *current))
    assert result.period == "2025"
    assert result.value == "12.00"
    assert result.citation.page_number == 2
