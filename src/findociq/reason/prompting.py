"""Prompt resource loading and deterministic evidence formatting."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING

from findociq.retrieve.schema import RetrievalHit

if TYPE_CHECKING:
    from findociq.reason.schema import Pass1Extraction


def load_prompt(name: str) -> str:
    return files("findociq.reason.prompts").joinpath(name).read_text(encoding="utf-8")


def render_evidence(hits: tuple[RetrievalHit, ...]) -> str:
    sections: list[str] = []
    for hit in hits:
        citations = [item.model_dump(mode="json") for item in hit.chunk.provenance]
        sections.append(
            json.dumps(
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "rank": hit.rank,
                    "score": hit.score,
                    "text": hit.chunk.text,
                    "provenance": citations,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(sections)


def render_extraction(extraction: Pass1Extraction) -> str:
    payload = {
        "question": extraction.question,
        "figures": [
            {
                "evidence_id": f"evidence_{index}",
                "label": figure.label,
                "value": figure.value,
                "unit": figure.unit,
                "period": figure.period,
            }
            for index, figure in enumerate(extraction.figures, start=1)
        ],
        "notes": extraction.notes,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def substitute(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered
