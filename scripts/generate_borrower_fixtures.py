"""Generate invented PDF acceptance fixtures. No borrower data is used."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Table, TableStyle

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "borrower"


def text_pdf(name, lines):
    canvas = Canvas(str(ROOT / name), pagesize=A4, invariant=1)
    canvas.setFont("Helvetica", 11)
    for index, line in enumerate(lines):
        canvas.drawString(48, 780 - index * 23, line)
    canvas.save()


def financial(year):
    canvas = Canvas(str(ROOT / f"financial-{year}.pdf"), pagesize=A4, invariant=1)
    canvas.setFont("Helvetica", 11)
    for index, line in enumerate(
        [
            "Audited financial statements - synthetic test fixture",
            "Legal name: Example Manufacturing LLP",
            "Standalone financial statements",
            f"Year ended FY{year - 1}-{str(year)[2:]}",
            "These figures are invented for software acceptance testing.",
        ]
    ):
        canvas.drawString(48, 780 - index * 23, line)
    canvas.showPage()
    canvas.setFont("Helvetica", 11)
    canvas.drawString(48, 805, "Statement of profit and loss and key ratios")
    canvas.drawString(48, 780, "Amounts in lakhs")
    rows = [
        ["Particulars", f"FY{year - 1}-{str(year)[2:]}", f"FY{year - 2}-{str(year - 1)[2:]}"],
        ["Revenue from operations", "1200", "900"],
        ["Profit after tax", "120", "90"],
        ["EBITDA margin", "18 %", "15 %"],
        ["DSCR", "1.8", "1.6"],
        ["Debt to equity ratio", "0.7", "0.8"],
        ["Debt to EBITDA", "1.2", "1.3"],
        ["Collateral cover", "1.5", "1.4"],
    ]
    table = Table(rows, colWidths=[280, 100, 100], rowHeights=30)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    table.wrapOn(canvas, 480, 600)
    table.drawOn(canvas, 48, 470)
    canvas.save()


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    text_pdf(
        "registration.pdf",
        [
            "Udyam registration certificate - synthetic",
            "Legal name: Example Manufacturing LLP",
            "Industry: Manufacturing",
            "Major activity: Manufacture of rubber and plastics products",
            "Total employees: 200",
            "Years operating: 10",
            "Region: West",
        ],
    )
    text_pdf(
        "gst.pdf",
        [
            "GST REG-06",
            "GST registration certificate - synthetic",
            "Legal name: Example Manufacturing LLP",
        ],
    )
    text_pdf(
        "agreement.pdf",
        [
            "LLP agreement - synthetic",
            "Legal name: Example Manufacturing LLP",
            "This document is not a financial statement.",
        ],
    )
    financial(2024)
    financial(2025)
