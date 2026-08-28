from urllib.parse import urlparse

import pytest

from findociq.ingest.gpu_lease import GpuLeaseConfig, VllmGpuLease


class FakeVllm:
    def __init__(self) -> None:
        self.sleeping = False
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request, timeout_seconds):  # type: ignore[no-untyped-def]
        del timeout_seconds
        parsed = urlparse(request.full_url)
        self.calls.append((request.method, f"{parsed.path}?{parsed.query}".rstrip("?")))
        if parsed.path == "/sleep":
            self.sleeping = True
            return None
        if parsed.path == "/wake_up":
            self.sleeping = False
            return None
        if parsed.path == "/is_sleeping":
            return self.sleeping
        if parsed.path == "/health":
            return None
        raise AssertionError(parsed.path)


def test_batch_sleeps_once_and_always_wakes() -> None:
    server = FakeVllm()
    lease = VllmGpuLease(GpuLeaseConfig(enabled=True), requester=server)

    with lease.ingestion_batch():
        assert server.sleeping is True

    assert server.sleeping is False
    assert server.calls == [
        ("POST", "/sleep?level=1"),
        ("GET", "/is_sleeping"),
        ("POST", "/wake_up"),
        ("GET", "/is_sleeping"),
        ("GET", "/health"),
    ]


def test_batch_failure_restores_vllm() -> None:
    server = FakeVllm()
    lease = VllmGpuLease(GpuLeaseConfig(enabled=True), requester=server)

    with pytest.raises(ValueError, match="bad PDF"), lease.ingestion_batch():
        raise ValueError("bad PDF")

    assert server.sleeping is False


def test_vision_fallback_temporarily_wakes_then_resleeps() -> None:
    server = FakeVllm()
    lease = VllmGpuLease(GpuLeaseConfig(enabled=True), requester=server)

    with lease.ingestion_batch():
        with lease.generation_fallback():
            assert server.sleeping is False
        assert server.sleeping is True

    assert server.sleeping is False


def test_gpu_lease_rejects_remote_control_endpoint() -> None:
    with pytest.raises(ValueError, match="local or internal"):
        GpuLeaseConfig(enabled=True, base_url="https://example.com")
