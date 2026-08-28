from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from fundermatch.orchestration.graph import (
    ApplicationMemoryGraph,
    WorkerContext,
    WorkerFailure,
)
from fundermatch.orchestration.lifecycle import InMemoryLifecycleStore
from fundermatch.orchestration.schema import (
    ApplicationMemoryState,
    GraphStatus,
    WorkerName,
    WorkerResult,
    result_hash,
)


@dataclass
class InjectedWorker:
    name: WorkerName
    fail_once: bool = False
    side_effect_before_failure: bool = False
    calls: int = 0
    committed_commands: set[str] = field(default_factory=set)

    async def run(
        self, state: ApplicationMemoryState, context: WorkerContext
    ) -> WorkerResult:
        self.calls += 1
        command = str(context.command_id)
        if self.side_effect_before_failure and command in self.committed_commands:
            return WorkerResult(
                worker=self.name,
                output_sha256=result_hash(self.name, state.application_id, command),
            )
        if self.fail_once and self.calls == 1:
            if self.side_effect_before_failure:
                self.committed_commands.add(command)
            raise WorkerFailure("injected_dependency_outage", "Injected safe failure")
        self.committed_commands.add(command)
        return WorkerResult(
            worker=self.name,
            output_sha256=result_hash(self.name, state.application_id, command),
        )


@pytest.mark.parametrize("failed_worker", tuple(WorkerName))
def test_resume_is_correct_at_every_worker_failure_boundary(
    failed_worker: WorkerName,
) -> None:
    async def scenario() -> None:
        workers = tuple(
            InjectedWorker(name, fail_once=name == failed_worker) for name in WorkerName
        )
        graph = ApplicationMemoryGraph(
            workers=workers,
            checkpointer=InMemorySaver(),
            lifecycle=InMemoryLifecycleStore(),
        )
        application_id = f"APP-FAIL-{failed_worker.value}"
        failed = await graph.start(application_id)
        assert failed.status == GraphStatus.FAILED_RETRYABLE
        assert failed.current_node == failed_worker
        resumed = await graph.resume(application_id)
        assert resumed.status == GraphStatus.WAITING_FOR_REVIEW
        for worker in workers:
            assert worker.calls == (2 if worker.name == failed_worker else 1)
            assert len(worker.committed_commands) == 1

    asyncio.run(scenario())


def test_side_effect_before_checkpoint_commit_is_not_repeated() -> None:
    async def scenario() -> None:
        worker = InjectedWorker(
            WorkerName.PRECEDENT_RETRIEVAL,
            fail_once=True,
            side_effect_before_failure=True,
        )
        graph = ApplicationMemoryGraph(
            workers=(worker,),
            checkpointer=InMemorySaver(),
            lifecycle=InMemoryLifecycleStore(),
        )
        failed = await graph.start("APP-SIDE-EFFECT-001")
        assert failed.status == GraphStatus.FAILED_RETRYABLE
        resumed = await graph.resume("APP-SIDE-EFFECT-001")
        assert resumed.status == GraphStatus.WAITING_FOR_REVIEW
        assert worker.calls == 2
        assert len(worker.committed_commands) == 1

    asyncio.run(scenario())
