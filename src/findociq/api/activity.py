"""Thread-safe, bounded activity registry for live local ingestion progress."""

from __future__ import annotations

from threading import Lock
from typing import Literal

from findociq.api.schema import IngestionActivityEvent, IngestionActivityResponse

ActivityStatus = Literal["pending", "running", "completed", "failed"]


class IngestionActivityRegistry:
    """Hold sanitized progress only; document contents never enter this registry."""

    def __init__(self, *, max_batches: int = 100, max_events_per_batch: int = 1000) -> None:
        self.max_batches = max_batches
        self.max_events_per_batch = max_events_per_batch
        self._lock = Lock()
        self._events: dict[str, list[IngestionActivityEvent]] = {}
        self._statuses: dict[str, ActivityStatus] = {}
        self._sequences: dict[str, int] = {}

    def start(self, batch_id: str) -> None:
        with self._lock:
            if batch_id not in self._events and len(self._events) >= self.max_batches:
                oldest = next(iter(self._events))
                self._events.pop(oldest, None)
                self._statuses.pop(oldest, None)
                self._sequences.pop(oldest, None)
            self._events[batch_id] = []
            self._statuses[batch_id] = "running"
            self._sequences[batch_id] = 0

    def report(self, batch_id: str, stage: str, message: str, **details: object) -> None:
        with self._lock:
            events = self._events.setdefault(batch_id, [])
            self._statuses.setdefault(batch_id, "running")
            sequence = self._sequences.get(batch_id, 0) + 1
            self._sequences[batch_id] = sequence
            event = IngestionActivityEvent(
                sequence=sequence,
                batch_id=batch_id,
                stage=stage,
                message=message,
                **details,
            )
            events.append(event)
            if len(events) > self.max_events_per_batch:
                del events[: len(events) - self.max_events_per_batch]

    def finish(self, batch_id: str, status: Literal["completed", "failed"]) -> None:
        with self._lock:
            self._statuses[batch_id] = status

    def snapshot(self, batch_id: str, *, after: int = 0) -> IngestionActivityResponse | None:
        with self._lock:
            if batch_id not in self._events:
                return None
            events = tuple(event for event in self._events[batch_id] if event.sequence > after)
            last = self._events[batch_id][-1].sequence if self._events[batch_id] else 0
            return IngestionActivityResponse(
                batch_id=batch_id,
                status=self._statuses.get(batch_id, "pending"),
                events=events,
                last_sequence=last,
            )
