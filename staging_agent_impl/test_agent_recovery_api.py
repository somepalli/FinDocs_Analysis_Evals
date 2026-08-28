from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from fundermatch.api.app import create_app
from fundermatch.intake_jobs import InMemoryIntakeJobStore
from fundermatch.orchestration.runtime import MemoryStatus
from fundermatch.orchestration.schema import ApplicationMemoryState, GraphStatus, WorkerName
from fundermatch.workflow.repository import InMemoryWorkflowRepository
from fundermatch.workflow.schema import ActorClaims, ActorRole


class RoleAuthenticator:
    def authenticate(self, token: str) -> ActorClaims:
        role = (
            ActorRole.PIPELINE if token == "pipeline-token" else ActorRole.HUMAN_REVIEWER
        )
        return ActorClaims(
            actor_id=f"{role.value}-test",
            display_name="Test Actor",
            roles=frozenset({role}),
        )


class RecoveryRuntime:
    def __init__(self, jobs: InMemoryIntakeJobStore) -> None:
        self.jobs = jobs
        self.graph = self
        self.resume_calls = 0
        self.cancel_calls = 0
        self._state = ApplicationMemoryState.initial(
            "APP-RECOVERY-API", job_id="job-recovery-api"
        ).model_copy(
            update={
                "status": GraphStatus.FAILED_RETRYABLE,
                "current_node": WorkerName.DOCUMENT_PROCESSING,
            }
        )

    async def state(self, application_id: str) -> ApplicationMemoryState:
        if application_id != self._state.application_id:
            raise KeyError(application_id)
        return self._state

    async def resume(self, job_id: str) -> ApplicationMemoryState:
        assert job_id == "job-recovery-api"
        self.resume_calls += 1
        if self._state.status == GraphStatus.FAILED_RETRYABLE:
            self._state = self._state.model_copy(
                update={"status": GraphStatus.WAITING_FOR_REVIEW}
            )
        return self._state

    async def cancel(self, job_id: str) -> ApplicationMemoryState:
        assert job_id == "job-recovery-api"
        self.cancel_calls += 1
        self._state = self._state.model_copy(update={"status": GraphStatus.CANCELLED})
        return self._state

    async def memory(self, application_id: str) -> MemoryStatus:
        state = await self.state(application_id)
        return MemoryStatus(
            application_id=state.application_id,
            job_id=state.job_id,
            status=state.status,
            current_worker=state.current_node,
            attempt=state.attempts.get(state.current_node, 0),
            completed_workers=state.completed_workers,
            pending_workers=state.pending_workers,
            last_error_code=None,
            last_error_retryable=None,
            checkpoint_time=datetime.now(UTC),
        )


def test_recovery_endpoints_are_sanitized_idempotent_and_role_protected() -> None:
    store = InMemoryIntakeJobStore()
    asyncio.run(store.create("job-recovery-api", "APP-RECOVERY-API"))
    runtime = RecoveryRuntime(store)
    app = create_app(
        InMemoryWorkflowRepository(),
        RoleAuthenticator(),
        intake_job_store=store,
        agent_runtime=runtime,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer pipeline-token"}
        first = client.post("/v1/intake-jobs/job-recovery-api/resume", headers=headers)
        repeated = client.post("/v1/intake-jobs/job-recovery-api/resume", headers=headers)
        assert first.status_code == repeated.status_code == 200
        assert first.json()["status"] == "waiting_for_review"
        assert "checkpoint" not in first.json()

        denied = client.post(
            "/v1/intake-jobs/job-recovery-api/cancel",
            headers={"Authorization": "Bearer reviewer-token"},
        )
        assert denied.status_code == 403
        cancelled = client.post("/v1/intake-jobs/job-recovery-api/cancel", headers=headers)
        repeated_cancel = client.post(
            "/v1/intake-jobs/job-recovery-api/cancel", headers=headers
        )
        assert cancelled.json()["status"] == "cancelled"
        assert repeated_cancel.json() == cancelled.json()

        memory = client.get(
            "/v1/applications/APP-RECOVERY-API/memory", headers=headers
        )
        assert memory.status_code == 200
        assert set(memory.json()) == {
            "application_id",
            "job_id",
            "status",
            "current_worker",
            "attempt",
            "completed_workers",
            "pending_workers",
            "last_error_code",
            "last_error_retryable",
            "checkpoint_time",
        }
        absent = client.get("/v1/applications/APP-OTHER/memory", headers=headers)
        assert absent.status_code == 404
