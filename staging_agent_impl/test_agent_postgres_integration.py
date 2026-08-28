from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import asyncpg
import pytest

from fundermatch.orchestration.graph import (
    ApplicationMemoryGraph,
    WorkerContext,
    WorkerFailure,
)
from fundermatch.orchestration.lifecycle import PostgresLifecycleStore
from fundermatch.orchestration.postgres import open_checkpointer
from fundermatch.orchestration.schema import (
    ApplicationMemoryState,
    GraphStatus,
    WorkerName,
    WorkerResult,
    result_hash,
)


@dataclass
class RestartWorker:
    name: WorkerName
    fail: bool = False
    calls: int = 0

    async def run(
        self, state: ApplicationMemoryState, context: WorkerContext
    ) -> WorkerResult:
        self.calls += 1
        if self.fail:
            raise WorkerFailure("restart_injected", "Injected restart boundary")
        return WorkerResult(
            worker=self.name,
            output_sha256=result_hash(self.name, state.application_id),
        )


@pytest.mark.skipif(
    "FUNDERMATCH_DATABASE_URL" not in os.environ,
    reason="PostgreSQL integration DSN is not configured",
)
def test_postgres_checkpoint_resumes_after_graph_process_recreation() -> None:
    async def scenario() -> None:
        dsn = os.environ["FUNDERMATCH_DATABASE_URL"]
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
        lifecycle = PostgresLifecycleStore(pool)
        application_id = "APP-PG-RESTART-CI"
        try:
            await pool.execute(
                "DELETE FROM langgraph.memory_threads WHERE thread_id = $1",
                application_id,
            )
            async with open_checkpointer(dsn) as first_saver:
                document = RestartWorker(WorkerName.DOCUMENT_PROCESSING)
                finance = RestartWorker(WorkerName.FINANCIAL_ANALYSIS, fail=True)
                first = ApplicationMemoryGraph(
                    workers=(document, finance),
                    checkpointer=first_saver,
                    lifecycle=lifecycle,
                )
                failed = await first.start(application_id)
                assert failed.status == GraphStatus.FAILED_RETRYABLE
                assert document.calls == 1

            async with open_checkpointer(dsn) as second_saver:
                new_document = RestartWorker(WorkerName.DOCUMENT_PROCESSING)
                new_finance = RestartWorker(WorkerName.FINANCIAL_ANALYSIS)
                second = ApplicationMemoryGraph(
                    workers=(new_document, new_finance),
                    checkpointer=second_saver,
                    lifecycle=lifecycle,
                )
                resumed = await second.resume(application_id)
                assert resumed.status == GraphStatus.WAITING_FOR_REVIEW
                assert new_document.calls == 0
                assert new_finance.calls == 1
                await second.mark_terminal(application_id, GraphStatus.CANCELLED)
                await second.checkpointer.adelete_thread(application_id)
            await lifecycle.delete(application_id)
        finally:
            await pool.close()

    asyncio.run(scenario())
