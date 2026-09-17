"""Exercise the adapter boundary, including Docling's parsed-page opt-in."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

from findociq.ingest.docling_parser import DocumentParser, ParserConfig
from findociq.ingest.schema import PageRoute


def test_adapter_retains_cells_and_restores_ocr_lines(monkeypatch):
    def box(top, bottom):
        return NS(l=10, r=180, t=top, b=bottom, coord_origin="TOPLEFT")

    cells = [
        NS(text=text, from_ocr=True, rect=NS(to_bounding_box=lambda y=y: box(y, y + 10)))
        for text, y in (("Legal name: Example LLP", 10), ("Industry: Manufacturing", 30))
    ]
    item = NS(
        label="text",
        text="Legal name: Example LLP Industry: Manufacturing",
        prov=[NS(page_no=1, bbox=box(10, 40))],
    )

    class Converter:
        def __init__(self, *, format_options):
            self.options = format_options["pdf"].pipeline_options

        def convert(self, path):
            # Reproduce real Docling cleanup: cells vanish without this flag.
            parsed = NS(textline_cells=cells) if self.options.generate_parsed_pages else None
            return NS(
                pages=[NS(page_no=0, parsed_page=parsed)],
                document=NS(
                    iterate_items=lambda: [(item, 0)],
                    pages={1: NS(size=NS(width=200, height=300))},
                ),
            )

    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.accelerator_options",
        NS(AcceleratorDevice=lambda value: value, AcceleratorOptions=lambda **kw: NS(**kw)),
    )
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", NS(InputFormat=NS(PDF="pdf")))
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.pipeline_options",
        NS(
            PdfPipelineOptions=lambda **kw: NS(
                generate_parsed_pages=kw.pop("generate_parsed_pages", False), **kw
            ),
            RapidOcrOptions=lambda **kw: NS(**kw),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.document_converter",
        NS(DocumentConverter=Converter, PdfFormatOption=lambda **kw: NS(**kw)),
    )
    result = DocumentParser(ParserConfig(accelerator_device="cpu"))._try_docling(
        Path("synthetic.pdf"),
        "synthetic",
        (NS(page_number=1, route=PageRoute.SCANNED),),
        timeout_seconds=30,
    )
    assert result is not None
    assert result.pages[0].blocks[0].text == "Legal name: Example LLP\nIndustry: Manufacturing"
