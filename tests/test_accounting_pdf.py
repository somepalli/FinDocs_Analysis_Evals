"""Invented accounting layout through real PDF parsing, not hand-built chunks."""

from decimal import Decimal
from pathlib import Path

import fitz

from findociq.ingest.chunker import LayoutAwareChunker
from findociq.ingest.docling_parser import DocumentParser, ParserConfig
from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidencePolicy


def create_accounting_pdf(path):
    document = fitz.open()
    page = document.new_page()
    for index, line in enumerate(
        (
            "Audited financial statements - invented acceptance fixture",
            "Legal name: Example Manufacturing LLP",
            "Standalone financial statements",
            "Year ended FY2024-25",
        )
    ):
        page.insert_text((40, 50 + index * 25), line, fontsize=11)
    page = document.new_page()
    page.insert_text((40, 40), "Statement of profit and loss and balance sheet", fontsize=11)
    page.insert_text((40, 65), "Amounts in lakhs", fontsize=11)
    rows = [
        ["", "Particulars", "Note", "FY2024-25", "FY2023-24"],
        ["1", "Revenue from operations", "20", "1000", "800"],
        ["8", "Profit/(Loss) before tax (6-7)", "", "100", "80"],
        ["", "Finance costs", "25", "20", "15"],
        ["", "Depreciation and amortisation expense", "26", "10", "8"],
        ["", "Net tax expense / (benefit)", "", "25", "20"],
        ["a.", "Long term borrowings", "5", "200", "180"],
        ["a.", "Short term borrowings", "8", "100", "90"],
        ["i.", "Partners' Contribution", "3", "100", "90"],
        ["ii.", "Partners' Current Account", "4", "200", "180"],
        ["", "Cash flow available for debt service", "", "90", "80"],
        ["", "Total debt service", "", "45", "40"],
    ]
    xs = [35, 65, 345, 385, 465, 550]
    for i, row in enumerate(rows):
        y = 100 + i * 30
        for j, value in enumerate(row):
            page.draw_rect(fitz.Rect(xs[j], y, xs[j + 1], y + 30))
            page.insert_text((xs[j] + 3, y + 18), value, fontsize=9)
    document.save(path)
    document.close()


def test_component_calculations_from_real_pdf(tmp_path):
    path = tmp_path / "invented-accounting.pdf"
    create_accounting_pdf(path)
    parsed = DocumentParser(ParserConfig(prefer_docling=False)).parse(path)
    chunks = LayoutAwareChunker().chunk(parsed)
    # Preserve this fixture's explicit CFADS policy and financial expectations.
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml")).model_dump()
    policy["calculations"]["dscr"] = "dscr_cfads_v1"
    policy["calculation_alternatives"]["dscr"] = []
    gate = BorrowerEvidenceGate(EvidencePolicy.model_validate(policy))
    expected = {
        "annual_revenue_crore": "10",
        "pat_crore": "0.75",
        "ebitda_margin_pct": "13",
        "debt_to_equity": "1",
        "dscr": "2",
    }
    for metric, value in expected.items():
        result = gate.extract(metric, chunks)
        assert Decimal(result.value) == Decimal(value), metric
        assert result.citation.page_number == 2
        assert result.evidence_validation.basis_resolution == "report_declaration"
    assert (
        gate.extract("debt_to_ebitda", chunks).calculation.formula_id
        == "debt_ebitda_borrowings_pbt_v1"
    )
