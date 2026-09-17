from findociq.ingest.ocr_lines import OcrLine, restore_lines
from findociq.ingest.schema import BoundingBox


def box(y, x=0):
    return BoundingBox(x0=x, y0=y, x1=x + 90, y1=y + 10)


def test_preserves_key_value_lines_from_flattened_docling_paragraph():
    cells = (
        OcrLine("Legal name:", box(10)),
        OcrLine("Example LLP", box(10, 100)),
        OcrLine("Industry: Manufacturing", box(30)),
    )
    block = BoundingBox(x0=0, y0=0, x1=200, y1=50)
    assert restore_lines("Legal name: Example LLP Industry: Manufacturing", block, cells) == (
        "Legal name: Example LLP\nIndustry: Manufacturing"
    )


def test_no_word_invention_or_cross_block_borrowing():
    original = "Legal name: Example LLP"
    assert restore_lines(original, box(10), (OcrLine("Other LLP", box(10)),)) == original
    assert restore_lines(original, box(10), (OcrLine(original, box(50)),)) == original
