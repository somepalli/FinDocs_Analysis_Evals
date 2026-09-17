"""Restore physical OCR lines without adding, deleting, or guessing source words."""

from dataclasses import dataclass

from findociq.ingest.schema import BoundingBox


@dataclass(frozen=True)
class OcrLine:
    text: str
    bbox: BoundingBox


def restore_lines(original: str, block: BoundingBox, cells: tuple[OcrLine, ...]) -> str:
    selected = sorted(
        (
            cell
            for cell in cells
            if (
                block.x0 <= (cell.bbox.x0 + cell.bbox.x1) / 2 <= block.x1
                and block.y0 <= (cell.bbox.y0 + cell.bbox.y1) / 2 <= block.y1
            )
        ),
        key=lambda cell: (cell.bbox.y0, cell.bbox.x0),
    )
    lines: list[list[OcrLine]] = []
    for cell in selected:
        if lines:
            previous = lines[-1][0].bbox
            overlap = min(previous.y1, cell.bbox.y1) - max(previous.y0, cell.bbox.y0)
            if overlap > 0.5 * min(previous.y1 - previous.y0, cell.bbox.y1 - cell.bbox.y0):
                lines[-1].append(cell)
                continue
        lines.append([cell])
    restored = "\n".join(
        " ".join(cell.text.strip() for cell in sorted(line, key=lambda cell: cell.bbox.x0))
        for line in lines
    )
    # Preserve the exact sequence of recognized tokens. Do not silently repair
    # missing OCR, multi-column reading-order differences, or unrelated cells.
    return restored if restored.split() == original.split() else original
