"""Docling-first parser with an explicit PyMuPDF digital-page fast path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import fitz

from findociq.ingest.router import PageRouter
from findociq.ingest.schema import (
    BlockType,
    BoundingBox,
    DocumentBlock,
    PageRoute,
    ParsedDocument,
    ParsedPage,
    Provenance,
    stable_document_id,
)
from findociq.ingest.vlm_fallback import (
    UnconfiguredGemmaVisionExtractor,
    VisionPageExtractor,
)


@dataclass(frozen=True, slots=True)
class ParserConfig:
    prefer_docling: bool = True
    fail_on_vision_required: bool = True
    accelerator_device: Literal["auto", "cpu", "cuda"] = "auto"
    ocr_backend: Literal["torch"] = "torch"


class TableExtractionFailed(RuntimeError):
    """Raised when the fast-path table detector cannot inspect a digital page."""


class DocumentParser:
    """Parse a PDF while preserving block geometry and page routing decisions."""

    def __init__(
        self,
        config: ParserConfig | None = None,
        router: PageRouter | None = None,
        vision_extractor: VisionPageExtractor | None = None,
    ) -> None:
        self.config = config or ParserConfig()
        self.router = router or PageRouter()
        self.vision_extractor = vision_extractor or UnconfiguredGemmaVisionExtractor()

    def parse(
        self,
        pdf_path: str | Path,
        *,
        timeout_seconds: float | None = None,
        max_pages: int | None = None,
    ) -> ParsedDocument:
        path = Path(pdf_path).resolve()
        deadline = monotonic() + timeout_seconds if timeout_seconds is not None else None
        document_id = stable_document_id(path)
        with fitz.open(path) as document:
            if max_pages is not None and document.page_count > max_pages:
                raise ValueError("document exceeds configured page limit")
        decisions = self.router.route(path)
        self._check_deadline(deadline)

        if self.config.prefer_docling and any(
            decision.route is not PageRoute.DIGITAL for decision in decisions
        ):
            docling_result = self._try_docling(
                path, document_id, decisions, timeout_seconds=timeout_seconds
            )
            if docling_result is not None:
                self._check_deadline(deadline)
                return docling_result

        pages: list[ParsedPage] = []
        with fitz.open(path) as document:
            for decision, page in zip(decisions, document, strict=True):
                self._check_deadline(deadline)
                if decision.route is PageRoute.DIGITAL:
                    try:
                        blocks = self._parse_digital_page(path, document_id, page)
                    except TableExtractionFailed:
                        blocks = self._vision_blocks(path, document_id, decision.page_number)
                else:
                    blocks = self._vision_blocks(path, document_id, decision.page_number)
                pages.append(
                    ParsedPage(
                        page_number=decision.page_number,
                        route=decision.route,
                        blocks=blocks,
                    )
                )
        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            pages=tuple(pages),
            parser_name="pymupdf-fast-path+gemma-vision",
        )

    @staticmethod
    def _check_deadline(deadline: float | None) -> None:
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError("document parsing exceeded the configured deadline")

    def _vision_blocks(
        self, path: Path, document_id: str, page_number: int
    ) -> tuple[DocumentBlock, ...]:
        try:
            return self.vision_extractor.extract_page(
                pdf_path=str(path),
                page_number=page_number,
                document_id=document_id,
            )
        except RuntimeError:
            if self.config.fail_on_vision_required:
                raise
            return ()

    def _try_docling(
        self,
        path: Path,
        document_id: str,
        decisions: tuple[Any, ...],
        *,
        timeout_seconds: float | None = None,
    ) -> ParsedDocument | None:
        """Use Docling when installed, falling back safely if unavailable.

        Docling's JSON schema changes more frequently than our internal schema.
        The adapter therefore consumes its stable document item iterator and
        returns `None` on an unsupported version rather than dropping geometry.
        """

        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError:
            return None

        try:
            device = AcceleratorDevice(self.config.accelerator_device)
            rapidocr_params = (
                {
                    "EngineConfig.torch.use_cuda": True,
                    "EngineConfig.torch.cuda_ep_cfg.device_id": 0,
                }
                if device is AcceleratorDevice.CUDA
                else {}
            )
            pipeline_options = PdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=device),
                document_timeout=timeout_seconds,
                ocr_options=RapidOcrOptions(
                    backend=self.config.ocr_backend,
                    rapidocr_params=rapidocr_params,
                ),
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            converted = converter.convert(path)
            docling_document = converted.document
            items = docling_document.iterate_items()
            by_page: dict[int, list[DocumentBlock]] = {
                decision.page_number: [] for decision in decisions
            }
            for order, entry in enumerate(items):
                item = entry[0] if isinstance(entry, tuple) else entry
                label = str(getattr(item, "label", "text")).lower()
                text = str(getattr(item, "text", "")).strip()
                if not text and "table" in label and hasattr(item, "export_to_markdown"):
                    text = str(item.export_to_markdown(docling_document)).strip()
                provenance_items = getattr(item, "prov", ())
                if not text or not provenance_items:
                    continue
                source = provenance_items[0]
                page_number = int(source.page_no)
                bbox = source.bbox
                page_size = docling_document.pages[page_number].size
                block_type = self._docling_block_type(label)
                by_page[page_number].append(
                    DocumentBlock(
                        block_type=block_type,
                        text=text,
                        provenance=Provenance(
                            document_id=document_id,
                            source_path=str(path),
                            page_number=page_number,
                            bbox=self._docling_bbox(bbox, float(page_size.height)),
                            page_width=float(page_size.width),
                            page_height=float(page_size.height),
                        ),
                        order=order,
                    )
                )
            if not any(by_page.values()):
                return None
            pages = tuple(
                ParsedPage(
                    page_number=decision.page_number,
                    route=decision.route,
                    blocks=tuple(by_page[decision.page_number]),
                )
                for decision in decisions
            )
            return ParsedDocument(
                document_id=document_id,
                source_path=str(path),
                pages=pages,
                parser_name="docling",
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _docling_block_type(label: str) -> BlockType:
        if "table" in label:
            return BlockType.TABLE
        if "caption" in label:
            return BlockType.CAPTION
        if "title" in label or "heading" in label or "section" in label:
            return BlockType.HEADING
        if "picture" in label or "image" in label:
            return BlockType.IMAGE
        return BlockType.PARAGRAPH

    @staticmethod
    def _docling_bbox(bbox: Any, page_height: float) -> BoundingBox:
        origin = str(getattr(bbox, "coord_origin", "")).lower()
        if "bottomleft" in origin:
            y0, y1 = page_height - float(bbox.t), page_height - float(bbox.b)
        else:
            y0, y1 = float(bbox.t), float(bbox.b)
        return BoundingBox(
            x0=min(float(bbox.l), float(bbox.r)),
            y0=min(y0, y1),
            x1=max(float(bbox.l), float(bbox.r)),
            y1=max(y0, y1),
        )

    @staticmethod
    def _parse_digital_page(
        path: Path,
        document_id: str,
        page: fitz.Page,
    ) -> tuple[DocumentBlock, ...]:
        table_rects = DocumentParser._table_rectangles(page)
        blocks: list[DocumentBlock] = []
        order = 0

        for table_index, table in enumerate(table_rects):
            markdown = table[1]
            blocks.append(
                DocumentBlock(
                    block_type=BlockType.TABLE,
                    text=markdown,
                    provenance=DocumentParser._provenance(path, document_id, page, table[0]),
                    order=order,
                    table_id=f"p{page.number + 1}-t{table_index + 1}",
                )
            )
            order += 1

        raw_blocks = page.get_text("blocks", sort=True)
        for raw in raw_blocks:
            rectangle = fitz.Rect(raw[:4])
            text = str(raw[4]).strip()
            if not text or any(rectangle.intersects(table[0]) for table in table_rects):
                continue
            blocks.append(
                DocumentBlock(
                    block_type=DocumentParser._infer_text_type(text),
                    text=text,
                    provenance=DocumentParser._provenance(path, document_id, page, rectangle),
                    order=order,
                )
            )
            order += 1

        blocks.sort(
            key=lambda block: (
                block.provenance.bbox.y0,
                block.provenance.bbox.x0,
                block.order,
            )
        )
        return tuple(
            block.model_copy(update={"order": index}) for index, block in enumerate(blocks)
        )

    @staticmethod
    def _table_rectangles(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
        try:
            tables = page.find_tables().tables
        except (AttributeError, RuntimeError) as error:
            raise TableExtractionFailed("PyMuPDF table extraction failed") from error
        results: list[tuple[fitz.Rect, str]] = []
        for table in tables:
            rows = table.extract()
            if not rows:
                continue
            normalized = [
                ["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row]
                for row in rows
            ]
            width = max(len(row) for row in normalized)
            padded = [row + [""] * (width - len(row)) for row in normalized]
            header = padded[0]
            separator = ["---"] * width
            markdown_rows = [header, separator, *padded[1:]]
            markdown = "\n".join("| " + " | ".join(row) + " |" for row in markdown_rows)
            results.append((fitz.Rect(table.bbox), markdown))
        return results

    @staticmethod
    def _infer_text_type(text: str) -> BlockType:
        first_line = text.splitlines()[0].strip()
        if first_line.lower().startswith(("table ", "figure ")) and len(text) < 240:
            return BlockType.CAPTION
        if len(first_line) < 100 and (first_line.isupper() or first_line.rstrip(":").istitle()):
            return BlockType.HEADING
        return BlockType.PARAGRAPH

    @staticmethod
    def _provenance(
        path: Path,
        document_id: str,
        page: fitz.Page,
        rectangle: fitz.Rect,
    ) -> Provenance:
        return Provenance(
            document_id=document_id,
            source_path=str(path),
            page_number=page.number + 1,
            bbox=BoundingBox.from_tuple(tuple(rectangle)),
            page_width=page.rect.width,
            page_height=page.rect.height,
        )
