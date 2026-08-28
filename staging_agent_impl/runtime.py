"""Intake-facing service around the durable application graph."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from fundermatch.intake import MAX_BATCH_BYTES, MAX_PDF_BYTES, IntakeMetadata
from fundermatch.intake_jobs import IntakeJob, IntakeJobStore
from fundermatch.orchestration.graph import ApplicationMemoryGraph
from fundermatch.orchestration.schema import ApplicationMemoryState, GraphStatus, WorkerName
from fundermatch.orchestration.workspace import ApplicationWorkspace


class MemoryStatus(BaseModel):
    """Sanitized recovery view. Checkpoint payloads are never returned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    application_id: str
    job_id: str | None
    status: GraphStatus
    current_worker: WorkerName
    attempt: int
    completed_workers: tuple[WorkerName, ...]
    pending_workers: tuple[WorkerName, ...]
    last_error_code: str | None
    last_error_retryable: bool | None
    checkpoint_time: datetime | None


class AgentActivityBridge:
    """Maps worker events to the existing content-safe intake activity stream."""

    def __init__(self, store: IntakeJobStore) -> None:
        self.store = store
        self._jobs: dict[str, str] = {}

    def bind(self, application_id: str, job_id: str) -> None:
        self._jobs[application_id] = job_id

    async def __call__(
        self, application_id: str, stage: str, **details: object
    ) -> None:
        job_id = self._jobs.get(application_id)
        if job_id is not None:
            await self.store.append(
                job_id,
                stage,
                f"Agent worker active: {stage.replace('_', ' ')}",
                **details,
            )


class AgentIntakeRuntime:
    def __init__(
        self,
        *,
        graph: ApplicationMemoryGraph,
        workspace: ApplicationWorkspace,
        jobs: IntakeJobStore,
        activity: AgentActivityBridge,
        max_file_bytes: int = MAX_PDF_BYTES,
        max_batch_bytes: int = MAX_BATCH_BYTES,
    ) -> None:
        self.graph = graph
        self.workspace = workspace
        self.jobs = jobs
        self.activity = activity
        self.max_file_bytes = max_file_bytes
        self.max_batch_bytes = max_batch_bytes
        self._application_locks: dict[str, asyncio.Lock] = {}

    async def start(
        self,
        metadata: IntakeMetadata,
        files: tuple[tuple[str, bytes], ...],
        *,
        job_id: str,
    ) -> ApplicationMemoryState:
        self.activity.bind(metadata.application_id, job_id)
        references = self.workspace.stage(
            metadata,
            files,
            max_file_bytes=self.max_file_bytes,
            max_batch_bytes=self.max_batch_bytes,
        )
        return await self._execute(
            job_id,
            self.graph.start(
                metadata.application_id,
                job_id=job_id,
                input_references=references,
            ),
        )

    async def resume(self, job_id: str) -> ApplicationMemoryState:
        job = await self.jobs.get(job_id)
        self.activity.bind(job.application_id, job_id)
        lock = self._application_locks.setdefault(job.application_id, asyncio.Lock())
        async with lock:
            current = await self.graph.state(job.application_id)
            if current.status not in {
                GraphStatus.FAILED_RETRYABLE,
                GraphStatus.NEEDS_ATTENTION,
            }:
                return current
            await self.jobs.append(
                job_id,
                "resume",
                "Resuming from the last successful agent checkpoint",
            )
            return await self._execute(job_id, self.graph.resume(job.application_id))

    async def cancel(self, job_id: str) -> ApplicationMemoryState:
        job = await self.jobs.get(job_id)
        lock = self._application_locks.setdefault(job.application_id, asyncio.Lock())
        async with lock:
            current = await self.graph.state(job.application_id)
            if current.status == GraphStatus.CANCELLED:
                return current
            if current.status.terminal:
                return current
            cancelled = await self.graph.mark_terminal(
                job.application_id, GraphStatus.CANCELLED
            )
            await self.jobs.append(job_id, "cancelled", "Agent processing was cancelled")
            await self.jobs.finish(job_id, status="cancelled")
            return cancelled

    async def memory(self, application_id: str) -> MemoryStatus:
        state = await self.graph.state(application_id)
        lifecycle = await self.graph.lifecycle.get(application_id)
        error = state.last_error
        return MemoryStatus(
            application_id=state.application_id,
            job_id=state.job_id,
            status=state.status,
            current_worker=state.current_node,
            attempt=state.attempts.get(state.current_node, 0),
            completed_workers=state.completed_workers,
            pending_workers=state.pending_workers,
            last_error_code=error.code if error else None,
            last_error_retryable=error.retryable if error else None,
            checkpoint_time=lifecycle.updated_at,
        )

    async def _execute(
        self,
        job_id: str,
        execution: Awaitable[ApplicationMemoryState],
    ) -> ApplicationMemoryState:
        state = await execution
        await self.jobs.append(
            job_id,
            "checkpoint",
            f"Checkpoint saved after {state.current_node.value}",
            retryable=state.last_error.retryable if state.last_error else None,
            error_code=state.last_error.code if state.last_error else None,
        )
        if state.status == GraphStatus.WAITING_FOR_REVIEW:
            await self.jobs.append(
                job_id,
                "waiting_for_review",
                "Agent flow paused for an authoritative human decision",
            )
            await self.jobs.finish(job_id, status="completed")
        elif state.status in {
            GraphStatus.FAILED_RETRYABLE,
            GraphStatus.NEEDS_ATTENTION,
            GraphStatus.FAILED_TERMINAL,
        }:
            await self.jobs.finish(
                job_id,
                status="failed",
                error_code=state.last_error.code if state.last_error else state.status.value,
                retryable=state.status == GraphStatus.FAILED_RETRYABLE,
            )
        return state
