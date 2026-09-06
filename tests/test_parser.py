from pathlib import Path
from types import SimpleNamespace

import pytest

from findociq.ingest.chunker import LayoutAwareChunker
from findociq.ingest.docling_parser import (
    DocumentParser,
    ParserConfig,
    TableExtractionFailed,
)
from findociq.ingest.schema import BlockType
from findociq.ingest.vlm_fallback import (
    OpenAICompatibleGemmaVisionExtractor,
    VisionConfig,
    VisionExtractionRequired,
)
from findociq.observability.recorder import InMemoryRecorder, TraceObserver


def test_vision_config_accepts_internal_vllm_service_endpoint() -> None:
    config = VisionConfig(
        base_url="http://vllm:8000/v1",
        model_id="google/gemma-3-4b-it",
        revision="8f28faf05c382a2dd81a471090acdb23156eb354",
    )
    assert config.base_url == "http://vllm:8000/v1"

FIXTURES = Path(__file__).parent / "fixtures"


def test_fast_path_preserves_page_and_bbox_provenance() -> None:
    parsed = DocumentParser(ParserConfig(prefer_docling=False)).parse(FIXTURES / "digital.pdf")
    assert parsed.parser_name.startswith("pymupdf")
    assert parsed.pages[0].blocks
    assert any(block.block_type is BlockType.TABLE for block in parsed.blocks)
    for block in parsed.blocks:
        source = block.provenance
        assert source.document_id == parsed.document_id
        assert source.page_number == 1
        assert 0 <= source.bbox.x0 <= source.bbox.x1 <= source.page_width
        assert 0 <= source.bbox.y0 <= source.bbox.y1 <= source.page_height


def test_scanned_page_never_silently_uses_text_dump() -> None:
    with pytest.raises(VisionExtractionRequired):
        DocumentParser(ParserConfig(prefer_docling=False)).parse(FIXTURES / "scanned.pdf")


def test_gemma_vision_returns_grounded_pdf_blocks() -> None:
    recorder = InMemoryRecorder()

    def requester(_request: object, _timeout: int) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"blocks":[{"block_type":"table","text":"| Revenue | 10 |",'
                            '"bbox":{"x0":100,"y0":200,"x1":900,"y1":600}}]}'
                        )
                    }
                }
            ]
        }

    extractor = OpenAICompatibleGemmaVisionExtractor(
        VisionConfig(
            base_url="http://localhost:8900/v1",
            model_id="gemma-vision",
            revision="pinned",
        ),
        TraceObserver(recorder),
        requester=requester,  # type: ignore[arg-type]
    )
    blocks = extractor.extract_page(
        pdf_path=str(FIXTURES / "scanned.pdf"), page_number=1, document_id="doc-vision"
    )
    assert len(blocks) == 1
    assert blocks[0].block_type is BlockType.TABLE
    assert blocks[0].table_id == "p1-vision-t1"
    assert blocks[0].provenance.document_id == "doc-vision"
    assert blocks[0].provenance.bbox.x0 == pytest.approx(
        blocks[0].provenance.page_width * 0.1
    )
    assert blocks[0].provenance.bbox.y1 == pytest.approx(
        blocks[0].provenance.page_height * 0.6
    )
    assert recorder.events[0].stage == "generation.vision"
    assert "data:image" not in recorder.events[0].model_dump_json()


def test_failed_fast_path_table_extraction_routes_to_gemma_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def requester(_request: object, _timeout: int) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"blocks":[{"block_type":"paragraph","text":"Recovered",'
                            '"bbox":{"x0":0,"y0":0,"x1":1000,"y1":1000}}]}'
                        )
                    }
                }
            ]
        }

    extractor = OpenAICompatibleGemmaVisionExtractor(
        VisionConfig(
            base_url="http://localhost:8900/v1",
            model_id="gemma-vision",
            revision="pinned",
        ),
        requester=requester,  # type: ignore[arg-type]
    )

    def fail_tables(_page: object) -> list[object]:
        raise TableExtractionFailed("table detector failed")

    monkeypatch.setattr(DocumentParser, "_table_rectangles", fail_tables)
    parsed = DocumentParser(
        ParserConfig(prefer_docling=False), vision_extractor=extractor
    ).parse(FIXTURES / "digital.pdf")
    assert parsed.blocks[0].text == "Recovered"
    assert parsed.blocks[0].metadata["extractor"] == "gemma3-vision"


def test_complex_page_attempts_docling_before_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted: list[bool] = []

    def fake_docling(*_args: object, **_kwargs: object) -> None:
        attempted.append(True)
        return None

    monkeypatch.setattr(DocumentParser, "_try_docling", fake_docling)
    with pytest.raises(VisionExtractionRequired):
        DocumentParser(ParserConfig(prefer_docling=True)).parse(FIXTURES / "scanned.pdf")
    assert attempted == [True]


def test_parser_enforces_configured_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_docling(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(DocumentParser, "_try_docling", slow_docling)
    with pytest.raises(TimeoutError, match="configured deadline"):
        DocumentParser(ParserConfig(prefer_docling=True)).parse(
            FIXTURES / "scanned.pdf", timeout_seconds=0.0
        )


def test_docling_bottom_left_boxes_convert_to_pdf_coordinates() -> None:
    bbox = SimpleNamespace(l=10, t=800, r=110, b=700, coord_origin="BOTTOMLEFT")
    converted = DocumentParser._docling_bbox(bbox, page_height=842)
    assert converted.model_dump() == {"x0": 10.0, "y0": 42.0, "x1": 110.0, "y1": 142.0}


def test_page_spanning_table_becomes_one_atomic_chunk() -> None:
    parsed = DocumentParser(ParserConfig(prefer_docling=False)).parse(
        FIXTURES / "page_spanning_table.pdf"
    )
    chunks = LayoutAwareChunker().chunk(parsed)
    tables = [chunk for chunk in chunks if chunk.kind == "table"]
    assert len(tables) == 1
    assert tables[0].metadata["atomic"] is True
    assert tables[0].metadata["page_spanning"] is True
    assert {item.page_number for item in tables[0].provenance} == {1, 2}
