"""Typed contracts shared across ingestion boundaries."""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    """A rectangle in PDF point coordinates."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def ordered_coordinates(self) -> BoundingBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox coordinates must satisfy x0 <= x1 and y0 <= y1")
        return self

    @classmethod
    def from_tuple(cls, value: tuple[float, float, float, float]) -> BoundingBox:
        return cls(x0=value[0], y0=value[1], x1=value[2], y1=value[3])

    def union(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


class Provenance(BaseModel):
    """Source evidence sufficient to render a chunk back onto a page."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source_path: str
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CAPTION = "caption"
    TABLE = "table"
    IMAGE = "image"


class PageRoute(StrEnum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"


class DocumentBlock(BaseModel):
    """A layout-preserving block emitted by a parser."""

    model_config = ConfigDict(frozen=True)

    block_type: BlockType
    text: str
    provenance: Provenance
    order: int = Field(ge=0)
    table_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    route: PageRoute
    blocks: tuple[DocumentBlock, ...]


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    source_path: str
    pages: tuple[ParsedPage, ...]
    parser_name: str

    @property
    def blocks(self) -> tuple[DocumentBlock, ...]:
        return tuple(block for page in self.pages for block in page.blocks)


class TextChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"
    chunk_id: str
    text: str
    provenance: tuple[Provenance, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["table"] = "table"
    chunk_id: str
    text: str
    table_text: str
    caption: str | None = None
    preceding_context: str | None = None
    provenance: tuple[Provenance, ...] = Field(min_length=1)
    table_provenance: tuple[Provenance, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


Chunk = Annotated[TextChunk | TableChunk, Field(discriminator="kind")]


def stable_document_id(path: str | Path) -> str:
    """Return a stable ID based on file bytes, independent of its location."""

    source = Path(path)
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(kind: str, text: str, provenance: tuple[Provenance, ...]) -> str:
    """Derive a deterministic chunk ID from content and source geometry."""

    evidence = "|".join(
        f"{item.document_id}:{item.page_number}:"
        f"{item.bbox.x0:.3f},{item.bbox.y0:.3f},{item.bbox.x1:.3f},{item.bbox.y1:.3f}"
        for item in provenance
    )
    return sha256(f"{kind}\n{text}\n{evidence}".encode()).hexdigest()
