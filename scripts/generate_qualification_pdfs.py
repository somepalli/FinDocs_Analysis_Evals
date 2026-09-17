"""Invented, image-only borrower PDFs for live GPU acceptance. Never use borrower files."""

import argparse
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Table, TableStyle


def generate(root: Path, pdftoppm: str, *, mixed_basis: bool = False) -> None:
    from pypdf import PdfReader

    root.mkdir(parents=True, exist_ok=True)
    identity = "Legal name: Example Manufacturing LLP"
    pages = {
        "registration.pdf": [
            [
                "Udyam registration certificate - synthetic",
                identity,
                "Industry: Auto components",
                "Major activity: Automotive parts",
                "Years operating: 10",
                "Region: North",
            ]
        ],
        "gst.pdf": [["GST registration certificate - synthetic", identity]],
        "agreement.pdf": [
            ["LLP agreement - synthetic", identity, "This document is not a financial statement."]
        ],
        "valuation.pdf": [
            [
                "Valuation report - synthetic",
                identity,
                "Year ended FY2024-25",
                "Collateral cover: 1.5",
            ]
        ],
        "financial.pdf": [
            [
                "Audited financial statements - synthetic",
                identity,
                "Standalone financial statements",
                "Year ended FY2024-25",
            ],
            ["Statement of profit and loss and cash flow", "Amounts in lakhs"],
        ],
    }
    if mixed_basis:
        pages["financial.pdf"].append(
            [
                "Consolidated financial statements",
                identity,
                "Year ended FY2024-25",
                "Separate statement section - synthetic reviewer routing fixture",
            ]
        )
    for name, sheets in pages.items():
        buffer = BytesIO()
        canvas = Canvas(buffer, pagesize=A4, invariant=1)
        for index, lines in enumerate(sheets):
            canvas.setFont("Helvetica", 13)
            for line_index, line in enumerate(lines):
                canvas.drawString(42, 790 - line_index * 25, line)
            if name == "financial.pdf" and index == 1:
                table = Table(
                    [
                        ["Particulars", "FY2024-25"],
                        ["Revenue from operations", "10000"],
                        ["EBITDA", "1250"],
                        ["Cash flow available for debt service", "150"],
                        ["Total debt service", "100"],
                        ["Total debt", "1500"],
                        ["Total equity", "3000"],
                    ],
                    colWidths=[370, 130],
                    rowHeights=35,
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 12),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ]
                    )
                )
                table.wrapOn(canvas, 500, 600)
                table.drawOn(canvas, 42, 455)
            canvas.showPage()
        canvas.save()
        # Rasterize to force genuine OCR; no hidden text layer is retained.
        with tempfile.TemporaryDirectory(prefix="synthetic-qualification-") as temporary:
            source = Path(temporary) / "digital.pdf"
            source.write_bytes(buffer.getvalue())
            subprocess.run(
                [pdftoppm, "-r", "144", "-png", str(source), str(Path(temporary) / "page")],
                check=True,
            )
            scanned = Canvas(str(root / name), pagesize=A4, invariant=1)
            for image in sorted(Path(temporary).glob("page-*.png")):
                scanned.drawImage(str(image), 0, 0, width=A4[0], height=A4[1])
                scanned.showPage()
            scanned.save()
        assert all(not p.extract_text().strip() for p in PdfReader(root / name).pages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--mixed-basis", action="store_true")
    args = parser.parse_args()
    generate(args.output, args.pdftoppm, mixed_basis=args.mixed_basis)
