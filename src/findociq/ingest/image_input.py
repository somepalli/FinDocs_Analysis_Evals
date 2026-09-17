"""Bounded image OCR adapter retaining original bytes and source identity."""

import warnings
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
from PIL import Image

from findociq.ingest.schema import stable_document_id

MAX_IMAGE_PIXELS = 25_000_000


def validate_image(content: bytes, suffix: str) -> None:
    expected = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}.get(suffix.lower())
    if expected is None:
        raise ValueError("unsupported_image_type")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format != expected or image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("image_format_or_pixel_limit")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("multi_frame_image_not_supported")
                image.verify()
        except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise ValueError("malformed_or_oversized_image") from error


def parse_image(path: Path, parse_pdf):
    validate_image(path.read_bytes(), path.suffix)
    # Same parent ensures encrypted-store tmpfs materialization stays on tmpfs.
    with TemporaryDirectory(prefix=".image-ocr-", dir=path.parent) as folder:
        with fitz.open(path) as image:
            wrapper = Path(folder) / "ocr.pdf"
            wrapper.write_bytes(image.convert_to_pdf())
        parsed = parse_pdf(wrapper)
    document_id = stable_document_id(path)
    pages = tuple(
        page.model_copy(
            update={
                "blocks": tuple(
                    block.model_copy(
                        update={
                            "provenance": block.provenance.model_copy(
                                update={
                                    "document_id": document_id,
                                    "source_path": str(path),
                                }
                            ),
                            "metadata": {
                                **block.metadata,
                                "source_media_type": path.suffix.lower(),
                                "coordinate_space": "single_page_image_pdf_points",
                                "image_adapter_version": "image-ocr-v1",
                            },
                        }
                    )
                    for block in page.blocks
                )
            }
        )
        for page in parsed.pages
    )
    return parsed.model_copy(
        update={
            "document_id": document_id,
            "source_path": str(path),
            "pages": pages,
            "parser_name": "image-ocr-v1+" + parsed.parser_name,
        }
    )
