from pathlib import Path

import fitz
from test_basis_context import page

from findociq.ingest.ruled_tables import activity_tables, dated_tables
from findociq.reason.evidence_gate import EvidencePolicy, profile


def test_dated_horizontal_rules_preserve_year_columns():
    with fitz.open() as pdf:
        sheet = pdf.new_page(width=600, height=800)
        rows = [
            ("Profit & Loss (Rs. Crore)", "31 Mar, 2023", "31 Mar, 2024", "31 Mar, 2025"),
            ("Net Revenue *", "10.00", "12.00", "15.00"),
            ("Profit for the Period", "1.00", "2.00", "3.00"),
            ("Finance Costs", "0.10", "0.20", "0.30"),
        ]
        for i, row in enumerate(rows):
            y = 70 + i * 24
            sheet.insert_text((35, y), row[0], fontsize=10)
            for right, text in zip((335, 455, 575), row[1:], strict=True):
                sheet.insert_text(
                    (right - fitz.get_text_length(text, fontsize=10), y), text, fontsize=10
                )
            sheet.draw_line((35, y + 6), (585, y + 6))
        sheet.draw_line((35, 54), (585, 54))
        tables = dated_tables(sheet)
        assert len(tables) == 1
        assert "| Net Revenue * | 10.00 | 12.00 | 15.00 |" in tables[0][1]
        assert "| Profit for the Period | 1.00 | 2.00 | 3.00 |" in tables[0][1]


def test_vendor_auditor_mentions_do_not_grant_audit_authority():
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    chunk = page(
        "EXAMPLE LIMITED\nThird-party financial information report\n"
        "Auditor's report\nGST registration\nStandalone financial data"
    )
    result = profile("report", (chunk,), policy)
    assert "financial_information_report" in result.types
    assert not result.types & {"audited_financials", "registration"}


def test_header_only_activity_table_recovers_body_with_original_box():
    with fitz.open() as pdf:
        page = pdf.new_page(width=700, height=800)
        edges = [30, 115, 270, 350, 590, 670]
        labels = [
            "Main Activity\nGroup Code",
            "Description of Main\nActivity Group",
            "Business\nActivity Code",
            "Description of Business Activity",
            "% of Turnover",
        ]
        for i, label in enumerate(labels):
            page.draw_rect(fitz.Rect(edges[i], 70, edges[i + 1], 100), fill=(0.8, 0.8, 0.8))
            page.insert_text((edges[i] + 2, 81), label, fontsize=8)
        values = ["Q", "Human health services", "86.00", "Health activities", "100.0"]
        for i, value in enumerate(values):
            page.insert_text((edges[i] + 2, 111), value, fontsize=8)
        page.draw_line((30, 119), (670, 119))
        page.draw_rect(fitz.Rect(30, 125, 670, 149), fill=(0, 0.3, 0.7))
        tables = activity_tables(page)
        assert len(tables) == 1
        assert "| Q | Human health services | 86.00 | Health activities | 100.0 |" in tables[0][1]
        assert tables[0][0].y1 == 119
