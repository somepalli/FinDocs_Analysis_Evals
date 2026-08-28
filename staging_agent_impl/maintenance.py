"""Scheduled stale-case flagging and terminal checkpoint cleanup."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict

from fundermatch.orchestration.graph import ApplicationMemoryGraph
from fundermatch.orchestration.lifecycle import MemoryStatusCounts


class MaintenanceResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stale_flagged: tuple[str, ...]
    terminal_deleted: tuple[str, ...]
    counts: MemoryStatusCounts


class AgentMaintenance:
    def __init__(
        self,
        graph: ApplicationMemoryGraph,
        *,
        stale_after: timedelta = timedelta(hours=24),
    ) -> None:
        self.graph = graph
        self.stale_after = stale_after

    async def run_once(self, now: datetime | None = None) -> MaintenanceResult:
        current = now or datetime.now(UTC)
        stale = await self.graph.flag_stale(current - self.stale_after)
        deleted = await self.graph.cleanup_expired(current)
        counts = await self.graph.lifecycle.counts()
        return MaintenanceResult(
            stale_flagged=stale,
            terminal_deleted=deleted,
            counts=counts,
        )
