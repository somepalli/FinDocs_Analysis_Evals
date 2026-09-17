"""Immutable application-scoped sets; membership must be authorized on every read."""

import json
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DocumentSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    application_id: str
    document_ids: tuple[str, ...] = Field(min_length=1, max_length=10000)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    set_id: str = Field(pattern=r"^[0-9a-f]{64}$")


def freeze(application_id: str, document_ids: tuple[str, ...], policy_hash: str) -> DocumentSet:
    ids = tuple(sorted(set(document_ids)))
    canonical = json.dumps([application_id, ids, policy_hash], separators=(",", ":"))
    return DocumentSet(
        application_id=application_id,
        document_ids=ids,
        policy_hash=policy_hash,
        set_id=sha256(canonical.encode()).hexdigest(),
    )


class DocumentSetStore:
    """SQLite for explicit local development; PostgreSQL for production."""

    def __init__(self, *, path: Path | None = None, dsn: str | None = None):
        self.path, self.dsn = path, dsn

    @contextmanager
    def _connection(self):
        if self.dsn:
            import psycopg

            with psycopg.connect(self.dsn) as connection:
                yield connection
            return
        if self.path is None:
            raise RuntimeError("document_set_store_unavailable")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS document_sets "
            "(set_id TEXT PRIMARY KEY, application_id TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def put(self, record: DocumentSet, owns: Callable[..., bool]) -> DocumentSet:
        if not owns(record.application_id, record.document_ids, policy_hash=record.policy_hash):
            raise PermissionError("evidence_scope_violation")
        table = "findociq.document_sets" if self.dsn else "document_sets"
        args = "%s, %s, %s" if self.dsn else "?, ?, ?"
        with self._connection() as connection:
            connection.execute(
                f"INSERT INTO {table} (set_id, application_id, payload) "
                f"VALUES ({args}) ON CONFLICT (set_id) DO NOTHING",
                (record.set_id, record.application_id, record.model_dump_json()),
            )
        return record

    def get(
        self, set_id: str, application_id: str, policy_hash: str, owns: Callable[..., bool]
    ) -> DocumentSet:
        table = "findociq.document_sets" if self.dsn else "document_sets"
        arg = "%s" if self.dsn else "?"
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT payload FROM {table} WHERE set_id={arg} AND application_id={arg}",
                (set_id, application_id),
            ).fetchone()
        if row is None:
            raise PermissionError("evidence_scope_violation")
        record = DocumentSet.model_validate_json(row[0])
        if (
            record.policy_hash != policy_hash
            or freeze(record.application_id, record.document_ids, record.policy_hash) != record
            or not owns(application_id, record.document_ids, policy_hash=policy_hash)
        ):
            raise PermissionError("evidence_scope_violation")
        return record
