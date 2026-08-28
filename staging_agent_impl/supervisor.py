"""Pinned Gemma router for ambiguous human send-back requests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fundermatch.orchestration.schema import WorkerName


class SupervisorRoutingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = "http://127.0.0.1:8900/v1"
    model_id: str = "google/gemma-3-12b-it"
    revision: str = "3b0c67b98eee8fb90633ef1bfbf3d39f43b9cf9d"
    temperature: Literal[0.0] = 0.0
    seed: int = 17
    max_tokens: int = Field(default=160, ge=64, le=500)
    min_confidence: float = Field(default=0.8, ge=0, le=1)
    prompt_path: Path = Path("prompts/sendback_supervisor_system.txt")


class SupervisorRoute(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: WorkerName | None
    confidence: float = Field(ge=0, le=1)
    ambiguous: bool


class GemmaSendBackRouter:
    def __init__(
        self,
        config: SupervisorRoutingConfig | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or SupervisorRoutingConfig()
        self._http_client = http_client

    async def route(self, reason: str) -> WorkerName | None:
        prompt = self.config.prompt_path.read_text(encoding="utf-8").strip()
        payload = {
            "model": self.config.model_id,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"reason": reason})},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "fundermatch_sendback_route",
                    "strict": True,
                    "schema": SupervisorRoute.model_json_schema(),
                },
            },
        }
        try:
            if self._http_client is not None:
                response = await self._http_client.post("/chat/completions", json=payload)
            else:
                async with httpx.AsyncClient(
                    base_url=self.config.base_url, timeout=120.0
                ) as client:
                    response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            route = SupervisorRoute.model_validate_json(content)
        except (httpx.HTTPError, KeyError, TypeError, ValidationError, OSError):
            return None
        if route.ambiguous or route.confidence < self.config.min_confidence:
            return None
        if route.stage in {None, WorkerName.HUMAN_REVIEW}:
            return None
        return route.stage
