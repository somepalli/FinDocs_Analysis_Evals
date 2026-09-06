from findociq.ingest.chunker import ChunkerConfig, LayoutAwareChunker
from findociq.ingest.schema import (
    BlockType,
    BoundingBox,
    DocumentBlock,
    PageRoute,
    ParsedDocument,
    ParsedPage,
    Provenance,
)


def evidence(page: int, y0: float, y1: float) -> Provenance:
    return Provenance(
        document_id="doc",
        source_path="filing.pdf",
        page_number=page,
        bbox=BoundingBox(x0=10, y0=y0, x1=500, y1=y1),
        page_width=595,
        page_height=842,
    )


def test_table_is_atomic_even_when_larger_than_text_limit() -> None:
    large_table = "| Metric | FY25 |\n" + "| Revenue | 100 |\n" * 100
    blocks = (
        DocumentBlock(
            block_type=BlockType.PARAGRAPH,
            text="The financial results are shown below.",
            provenance=evidence(1, 10, 30),
            order=0,
        ),
        DocumentBlock(
            block_type=BlockType.CAPTION,
            text="Table 1: Results",
            provenance=evidence(1, 35, 45),
            order=1,
        ),
        DocumentBlock(
            block_type=BlockType.TABLE,
            text=large_table,
            provenance=evidence(1, 50, 800),
            order=2,
            table_id="table-1",
        ),
    )
    document = ParsedDocument(
        document_id="doc",
        source_path="filing.pdf",
        pages=(ParsedPage(page_number=1, route=PageRoute.DIGITAL, blocks=blocks),),
        parser_name="test",
    )
    chunks = LayoutAwareChunker(
        ChunkerConfig(max_text_characters=100, overlap_characters=10)
    ).chunk(document)
    table_chunks = [chunk for chunk in chunks if chunk.kind == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].table_text == large_table
    assert table_chunks[0].caption == "Table 1: Results"
    assert table_chunks[0].preceding_context == "The financial results are shown below."
    assert table_chunks[0].metadata["atomic"] is True
    assert len(table_chunks[0].provenance) == 3


def test_text_splitting_is_deterministic_and_bounded() -> None:
    block = DocumentBlock(
        block_type=BlockType.PARAGRAPH,
        text="word " * 150,
        provenance=evidence(1, 10, 100),
        order=0,
    )
    document = ParsedDocument(
        document_id="doc",
        source_path="filing.pdf",
        pages=(ParsedPage(page_number=1, route=PageRoute.DIGITAL, blocks=(block,)),),
        parser_name="test",
    )
    chunker = LayoutAwareChunker(ChunkerConfig(max_text_characters=120, overlap_characters=20))
    first = chunker.chunk(document)
    second = chunker.chunk(document)
    assert first == second
    assert all(len(chunk.text) <= 120 for chunk in first)
    assert all(chunk.provenance for chunk in first)


def test_oversized_atomic_table_fails_closed_without_splitting() -> None:
    table = DocumentBlock(
        block_type=BlockType.TABLE,
        text="x" * 101,
        provenance=evidence(1, 10, 100),
        order=0,
        table_id="oversized",
    )
    document = ParsedDocument(
        document_id="doc",
        source_path="filing.pdf",
        pages=(ParsedPage(page_number=1, route=PageRoute.DIGITAL, blocks=(table,)),),
        parser_name="test",
    )
    chunker = LayoutAwareChunker(
        ChunkerConfig(
            max_text_characters=100,
            overlap_characters=10,
            max_table_characters=100,
        )
    )

    import pytest

    with pytest.raises(ValueError, match="atomic table"):
        chunker.chunk(document)
