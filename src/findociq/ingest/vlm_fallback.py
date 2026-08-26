"""Local Gemma 3 vision extraction for scanned and hybrid PDF pages."""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import fitz
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from findociq.ingest.schema import BlockType, BoundingBox, DocumentBlock, Provenance
from findociq.observability.recorder import TraceObserver
from findociq.observability.schema import TraceContext


class VisionExtractionRequired(RuntimeError):
    """Raised when a page needs vision but no open-weights extractor is set."""


class VisionConfig(BaseModel):
    """Typed deterministic settings for a local multimodal Gemma endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    backend: Literal["vllm", "ollama"] = "vllm"
    base_url: str
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    temperature: Literal[0.0] = 0.0
    seed: int = 17
    max_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: int = Field(default=180, gt=0)
    render_dpi: int = Field(default=144, ge=72, le=300)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VisionConfig:
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"vision config must be a mapping: {source}")
        return cls(**payload)

    @model_validator(mode="after")
    def validate_local_endpoint(self) -> VisionConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("vision base_url must be a local HTTP endpoint")
        return self


class NormalizedBox(BaseModel):
    """Model-emitted rectangle on a 0..1000 page coordinate system."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x0: float = Field(ge=0, le=1000)
    y0: float = Field(ge=0, le=1000)
    x1: float = Field(ge=0, le=1000)
    y1: float = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def ordered(self) -> NormalizedBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("normalized bbox coordinates must be ordered")
        return self


class VisionBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_type: BlockType
    text: str = Field(min_length=1)
    bbox: NormalizedBox


class VisionPageResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    blocks: tuple[VisionBlock, ...]


class VisionPageExtractor(ABC):
    @abstractmethod
    def extract_page(
        self,
        *,
        pdf_path: str,
        page_number: int,
        document_id: str,
    ) -> tuple[DocumentBlock, ...]:
        """Extract ordered, grounded blocks from one rendered page."""


VisionRequester = Callable[[Request, int], dict[str, Any]]


class OpenAICompatibleGemmaVisionExtractor(VisionPageExtractor):
    """Render one page and extract typed blocks through local Gemma 3 vision."""

    def __init__(
        self,
        config: VisionConfig,
        observer: TraceObserver | None = None,
        requester: VisionRequester | None = None,
    ) -> None:
        self.config = config
        self.observer = observer or TraceObserver()
        self.requester = requester or _request_json

    def extract_page(
        self,
        *,
        pdf_path: str,
        page_number: int,
        document_id: str,
    ) -> tuple[DocumentBlock, ...]:
        if not self.config.enabled:
            raise VisionExtractionRequired("Gemma 3 vision extraction is disabled")
        path = Path(pdf_path)
        with fitz.open(path) as document:
            if page_number > document.page_count:
                raise ValueError(f"page {page_number} does not exist in {path}")
            page = document[page_number - 1]
            page_width, page_height = page.rect.width, page.rect.height
            pixmap = page.get_pixmap(dpi=self.config.render_dpi, alpha=False)
            image_data = base64.b64encode(pixmap.tobytes("png")).decode("ascii")

        context = TraceContext.for_query(
            f"{document_id}:{page_number}", operation="ingestion:vision"
        )
        with self.observer.span(
            context,
            "generation.vision",
            {
                "backend": self.config.backend,
                "model_id": self.config.model_id,
                "model_revision": self.config.revision,
                "page_number": page_number,
                "render_dpi": self.config.render_dpi,
            },
        ) as attributes:
            response = self.requester(
                self._request(image_data, page_number), self.config.timeout_seconds
            )
            parsed = VisionPageResponse.model_validate_json(_response_content(response))
            attributes["block_count"] = len(parsed.blocks)

        blocks = [
            DocumentBlock(
                block_type=item.block_type,
                text=item.text.strip(),
                provenance=Provenance(
                    document_id=document_id,
                    source_path=str(path.resolve()),
                    page_number=page_number,
                    bbox=_pdf_box(item.bbox, page_width, page_height),
                    page_width=page_width,
                    page_height=page_height,
                ),
                order=index,
                table_id=f"p{page_number}-vision-t{index + 1}"
                if item.block_type is BlockType.TABLE
                else None,
                metadata={"extractor": "gemma3-vision"},
            )
            for index, item in enumerate(parsed.blocks)
        ]
        blocks.sort(
            key=lambda item: (
                item.provenance.bbox.y0,
                item.provenance.bbox.x0,
                item.order,
            )
        )
        return tuple(
            block.model_copy(update={"order": index}) for index, block in enumerate(blocks)
        )

    def _request(self, image_data: str, page_number: int) -> Request:
        prompt = (
            files("findociq.ingest.prompts")
            .joinpath("vision_extract.txt")
            .read_text(encoding="utf-8")
            .replace("{{page_number}}", str(page_number))
        )
        payload = {
            "model": self.config.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        return Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )


class UnconfiguredGemmaVisionExtractor(VisionPageExtractor):
    def extract_page(
        self,
        *,
        pdf_path: str,
        page_number: int,
        document_id: str,
    ) -> tuple[DocumentBlock, ...]:
        del document_id
        raise VisionExtractionRequired(
            f"page {page_number} of {pdf_path} requires Gemma 3 vision extraction; "
            "configure a local open-weights vision backend"
        )


def _request_json(request: Request, timeout_seconds: int) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read(2_000).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"local Gemma vision extraction failed: HTTP {error.code}: {detail}"
        ) from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"local Gemma vision extraction failed: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("local Gemma vision response must be a JSON object")
    return payload


def _response_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("local Gemma vision response did not contain message content") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("local Gemma vision returned empty message content")
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    return stripped


def _pdf_box(box: NormalizedBox, page_width: float, page_height: float) -> BoundingBox:
    return BoundingBox(
        x0=box.x0 * page_width / 1000,
        y0=box.y0 * page_height / 1000,
        x1=box.x1 * page_width / 1000,
        y1=box.y1 * page_height / 1000,
    )
