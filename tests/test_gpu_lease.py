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


def test_distinct_services_serialize_model_operations() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    ingestion = VllmGpuLease(GpuLeaseConfig())
    extraction = VllmGpuLease(GpuLeaseConfig())
    attempted, entered = Event(), Event()

    def concurrent_load() -> None:
        attempted.set()
        with extraction.operation():
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with ingestion.ingestion_batch():
            future = pool.submit(concurrent_load)
            assert attempted.wait(2)
            assert not entered.wait(0.1)
        future.result(timeout=2)
        assert entered.is_set()


@pytest.mark.parametrize("fail_retrieval", [False, True])
def test_query_releases_retrieval_before_waking_generation(fail_retrieval: bool) -> None:
    from findociq.service import FinDocIQService

    server = FakeVllm()
    events = []

    class Retrieval:
        def retrieve(self, *args, **kwargs):
            assert server.sleeping
            events.append("retrieve")
            if fail_retrieval:
                raise ValueError("retrieval failure")
            return ()

        def release_models(self):
            assert server.sleeping
            events.append("release")

    class Reasoning:
        def run(self, *args, **kwargs):
            assert not server.sleeping
            assert events == ["retrieve", "release"]
            events.append("reason")
            return "done"

    service = FinDocIQService(
        Retrieval(), {"two_pass": Reasoning()},
        gpu_lease=VllmGpuLease(GpuLeaseConfig(enabled=True), requester=server),
    )
    if fail_retrieval:
        with pytest.raises(ValueError, match="retrieval failure"):
            service.query("test", mode="two_pass")
        assert events == ["retrieve", "release"]
    else:
        assert service.query("test", mode="two_pass") == "done"
    assert not server.sleeping
