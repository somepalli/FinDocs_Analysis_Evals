"""Batch-scoped GPU ownership switching for local vLLM and ingestion models."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GpuLeaseConfig(BaseModel):
    """Typed local-only configuration for vLLM sleep-mode coordination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    base_url: str = "http://127.0.0.1:8900"
    sleep_level: Literal[1] = 1
    timeout_seconds: int = Field(default=600, gt=0, le=3600)

    @model_validator(mode="after")
    def require_local_endpoint(self) -> GpuLeaseConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
            "vllm",
        }:
            raise ValueError("gpu lease base_url must be a local or internal vLLM endpoint")
        return self


GpuRequester = Callable[[Request, int], Any]

# All in-process ingestion and extraction services share the same model runtime.
_MODEL_RUNTIME_LOCK = RLock()


class GpuLeaseError(RuntimeError):
    """vLLM could not safely transfer GPU ownership."""


class VllmGpuLease:
    """Serialize ingestion batches and restore vLLM even when a batch fails."""

    def __init__(
        self,
        config: GpuLeaseConfig,
        requester: GpuRequester | None = None,
    ) -> None:
        self.config = config
        self._requester = requester or _request
        self._lock = _MODEL_RUNTIME_LOCK
        self._batch_sleeping = False

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Serialize model initialization and inference across API services."""
        with self._lock:
            yield

    @contextmanager
    def ingestion_batch(self) -> Iterator[None]:
        """Give one complete ingestion batch exclusive use of the GPU."""

        if not self.config.enabled:
            with self._lock:
                yield
            return
        with self._lock:
            self._sleep()
            self._batch_sleeping = True
            try:
                yield
            except Exception as batch_error:
                try:
                    self._wake()
                except Exception as wake_error:
                    raise ExceptionGroup(
                        "ingestion failed and vLLM could not be restored",
                        [batch_error, wake_error],
                    ) from None
                raise
            else:
                self._wake()
            finally:
                self._batch_sleeping = False

    @contextmanager
    def generation_fallback(self) -> Iterator[None]:
        """Temporarily wake Gemma when Docling must fall back to vision."""

        if not self.config.enabled or not self._batch_sleeping:
            yield
            return
        self._wake()
        try:
            yield
        finally:
            self._sleep()

    def _sleep(self) -> None:
        self._call("POST", "/sleep", query={"level": str(self.config.sleep_level)})
        if self._sleep_state() is not True:
            raise GpuLeaseError("vLLM did not enter sleep mode")

    def _wake(self) -> None:
        self._call("POST", "/wake_up")
        if self._sleep_state() is not False:
            raise GpuLeaseError("vLLM did not leave sleep mode")
        self._call("GET", "/health")

    def _sleep_state(self) -> bool:
        payload = self._call("GET", "/is_sleeping")
        if isinstance(payload, bool):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("is_sleeping"), bool):
            return payload["is_sleeping"]
        raise GpuLeaseError("vLLM returned an invalid sleep-state response")

    def _call(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        query: dict[str, str] | None = None,
    ) -> Any:
        suffix = f"?{urlencode(query)}" if query else ""
        request = Request(
            self.config.base_url.rstrip("/") + path + suffix,
            data=b"" if method == "POST" else None,
            method=method,
        )
        try:
            return self._requester(request, self.config.timeout_seconds)
        except GpuLeaseError:
            raise
        except Exception as error:
            raise GpuLeaseError(f"vLLM GPU lifecycle request failed for {path}") from error


def _request(request: Request, timeout_seconds: int) -> Any:
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            body = response.read()
    except HTTPError as error:
        raise GpuLeaseError(
            f"vLLM GPU lifecycle endpoint returned HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError) as error:
        raise GpuLeaseError("vLLM GPU lifecycle endpoint is unavailable") from error
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise GpuLeaseError("vLLM GPU lifecycle endpoint returned invalid JSON") from error
