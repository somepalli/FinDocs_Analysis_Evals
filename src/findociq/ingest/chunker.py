"""Layout-aware chunking that treats tables as indivisible evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from findociq.ingest.schema import (
    BlockType,
    Chunk,
    DocumentBlock,
    ParsedDocument,
    Provenance,
    TableChunk,
    TextChunk,
    stable_chunk_id,
)


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    max_text_characters: int = 1_800
    overlap_characters: int = 180
    max_table_characters: int = 50_000

    def __post_init__(self) -> None:
        if self.max_text_characters <= 0:
            raise ValueError("max_text_characters must be positive")
        if not 0 <= self.overlap_characters < self.max_text_characters:
            raise ValueError("overlap_characters must be >= 0 and smaller than max")
        if self.max_table_characters <= 0:
            raise ValueError("max_table_characters must be positive")


class LayoutAwareChunker:
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(self, document: ParsedDocument) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        blocks = document.blocks
        last_table_position: int | None = None
        for index, block in enumerate(blocks):
            if block.block_type is BlockType.TABLE:
                table_chunk = self._table_chunk(blocks, index)
                if self._is_continuation(table_chunk) and last_table_position is not None:
                    position = int(last_table_position)
                    previous = chunks[position]
                    if isinstance(previous, TableChunk):
                        chunks[position] = self._merge_table_parts(previous, table_chunk)
                        continue
                chunks.append(table_chunk)
                last_table_position = len(chunks) - 1
            elif block.block_type not in {BlockType.CAPTION, BlockType.IMAGE}:
                chunks.extend(self._text_chunks(block))
        return tuple(chunks)

    def _table_chunk(self, blocks: tuple[DocumentBlock, ...], index: int) -> TableChunk:
        table = blocks[index]
        if len(table.text) > self.config.max_table_characters:
            raise ValueError("atomic table exceeds configured character limit")
        caption_block = self._adjacent(blocks, index, BlockType.CAPTION)
        preceding_block = self._preceding_paragraph(blocks, index, caption_block)
        caption = caption_block.text if caption_block else None
        preceding = preceding_block.text if preceding_block else None
        parts = [part for part in (preceding, caption, table.text) if part]
        text = "\n\n".join(parts)
        provenance = self._unique_provenance(
            item.provenance for item in (preceding_block, caption_block, table) if item is not None
        )
        return TableChunk(
            chunk_id=stable_chunk_id("table", text, provenance),
            text=text,
            table_text=table.text,
            caption=caption,
            preceding_context=preceding,
            provenance=provenance,
            metadata={"table_id": table.table_id, "atomic": True},
        )

    def _text_chunks(self, block: DocumentBlock) -> list[TextChunk]:
        text = block.text.strip()
        if not text:
            return []
        parts = self._split_text(text)
        provenance = (block.provenance,)
        return [
            TextChunk(
                chunk_id=stable_chunk_id("text", part, provenance),
                text=part,
                provenance=provenance,
                metadata={"block_type": block.block_type.value, "part": index},
            )
            for index, part in enumerate(parts)
        ]

    def _split_text(self, text: str) -> list[str]:
        maximum = self.config.max_text_characters
        overlap = self.config.overlap_characters
        if len(text) <= maximum:
            return [text]

        pieces: list[str] = []
        start = 0
        while start < len(text):
            upper = min(start + maximum, len(text))
            end = upper
            if upper < len(text):
                candidates = [
                    text.rfind("\n\n", start, upper),
                    text.rfind(". ", start, upper),
                    text.rfind(" ", start, upper),
                ]
                viable = [candidate for candidate in candidates if candidate > start]
                if viable:
                    end = max(viable)
                    if text[end : end + 2] == ". ":
                        end += 1
            if end <= start:
                end = upper
            pieces.append(text[start:end].strip())
            if end >= len(text):
                break
            next_start = max(0, end - overlap)
            if next_start <= start:
                next_start = end
            start = next_start
        return [piece for piece in pieces if piece]

    @staticmethod
    def _adjacent(
        blocks: tuple[DocumentBlock, ...], index: int, block_type: BlockType
    ) -> DocumentBlock | None:
        for candidate_index in (index - 1, index + 1):
            if 0 <= candidate_index < len(blocks):
                candidate = blocks[candidate_index]
                if (
                    candidate.block_type is block_type
                    and candidate.provenance.page_number == blocks[index].provenance.page_number
                ):
                    return candidate
        return None

    @staticmethod
    def _preceding_paragraph(
        blocks: tuple[DocumentBlock, ...],
        index: int,
        caption: DocumentBlock | None,
    ) -> DocumentBlock | None:
        start = index - 1
        if caption is not None and start >= 0 and blocks[start] is caption:
            start -= 1
        if start < 0:
            return None
        candidate = blocks[start]
        if candidate.block_type not in {BlockType.PARAGRAPH, BlockType.HEADING}:
            return None
        # The preceding paragraph may be on the prior page for a page-spanning
        # table, but never reach further back than that.
        if blocks[index].provenance.page_number - candidate.provenance.page_number > 1:
            return None
        return candidate

    @staticmethod
    def _unique_provenance(items: Iterable[Provenance]) -> tuple[Provenance, ...]:
        unique: list[Provenance] = []
        seen: set[str] = set()
        for item in items:
            key = item.model_dump_json()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return tuple(unique)

    @staticmethod
    def _is_continuation(chunk: TableChunk) -> bool:
        caption = (chunk.caption or "").casefold()
        return "continued" in caption or bool(chunk.metadata.get("continued"))

    def _merge_table_parts(self, first: TableChunk, continuation: TableChunk) -> TableChunk:
        table_text = f"{first.table_text}\n\n{continuation.table_text}"
        if len(table_text) > self.config.max_table_characters:
            raise ValueError("merged atomic table exceeds configured character limit")
        captions = [value for value in (first.caption, continuation.caption) if value]
        caption = " / ".join(captions) or None
        text = "\n\n".join(
            value for value in (first.preceding_context, caption, table_text) if value
        )
        provenance = LayoutAwareChunker._unique_provenance(
            (*first.provenance, *continuation.provenance)
        )
        metadata = dict(first.metadata)
        metadata.update(
            {
                "atomic": True,
                "page_spanning": True,
                "parts": int(metadata.get("parts", 1)) + 1,
            }
        )
        return TableChunk(
            chunk_id=stable_chunk_id("table", text, provenance),
            text=text,
            table_text=table_text,
            caption=caption,
            preceding_context=first.preceding_context,
            provenance=provenance,
            metadata=metadata,
        )
