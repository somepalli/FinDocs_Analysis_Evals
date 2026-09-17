from hashlib import sha256
from io import BytesIO

import pytest
from PIL import Image

from findociq.ingest.image_input import parse_image, validate_image
from findociq.ingest.schema import (
    BoundingBox,
    DocumentBlock,
    ParsedDocument,
    ParsedPage,
    Provenance,
)


def image_bytes():
    out = BytesIO()
    Image.new("RGB", (300, 200), "white").save(out, format="PNG")
    return out.getvalue()


def test_image_provenance_refers_to_original_and_wrapper_is_temporary(tmp_path):
    path = tmp_path / "proof.png"
    content = image_bytes()
    path.write_bytes(content)
    wrappers = []

    def parser(wrapper):
        wrappers.append(wrapper)
        block = DocumentBlock(
            block_type="paragraph",
            text="Invented incorporation proof",
            order=0,
            provenance=Provenance(
                document_id="temporary",
                source_path=str(wrapper),
                page_number=1,
                bbox=BoundingBox(x0=1, y0=1, x1=30, y1=20),
                page_width=225,
                page_height=150,
            ),
        )
        return ParsedDocument(
            document_id="temporary",
            source_path=str(wrapper),
            parser_name="test",
            pages=(ParsedPage(page_number=1, route="scanned", blocks=(block,)),),
        )

    result = parse_image(path, parser)
    assert result.document_id == sha256(content).hexdigest()
    assert result.blocks[0].provenance.document_id == result.document_id
    assert result.blocks[0].provenance.source_path == str(path)
    assert path.read_bytes() == content
    assert not wrappers[0].exists()


def test_image_format_mismatch_rejected():
    with pytest.raises(ValueError):
        validate_image(image_bytes(), ".jpeg")


def test_malformed_image_rejected_before_parser():
    with pytest.raises(ValueError):
        validate_image(b"\x89PNG\r\n\x1a\nbroken", ".png")
