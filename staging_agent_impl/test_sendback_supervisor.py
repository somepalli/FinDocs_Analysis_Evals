from __future__ import annotations

import asyncio
import json

import httpx

from fundermatch.orchestration.schema import WorkerName
from fundermatch.orchestration.supervisor import (
    GemmaSendBackRouter,
    SupervisorRoutingConfig,
)


def test_pinned_supervisor_routes_clear_reason_and_stops_ambiguous_reason(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return only the routing schema.", encoding="utf-8")
    responses = iter(
        (
            {"stage": "financial_analysis", "confidence": 0.97, "ambiguous": False},
            {"stage": None, "confidence": 0.4, "ambiguous": True},
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["temperature"] == 0.0
        assert payload["seed"] == 17
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(next(responses))}}
                ]
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://vllm/v1"
        ) as client:
            router = GemmaSendBackRouter(
                SupervisorRoutingConfig(prompt_path=prompt), http_client=client
            )
            assert (
                await router.route("The EBITDA margin uses the wrong period")
                == WorkerName.FINANCIAL_ANALYSIS
            )
            assert await router.route("Please take another look") is None

    asyncio.run(scenario())


def test_supervisor_dependency_failure_stops_without_guessing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return only the routing schema.", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://vllm/v1"
        ) as client:
            router = GemmaSendBackRouter(
                SupervisorRoutingConfig(prompt_path=prompt), http_client=client
            )
            assert await router.route("Rerun something") is None

    asyncio.run(scenario())
