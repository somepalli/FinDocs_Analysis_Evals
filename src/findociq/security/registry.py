"""Application-scoped document ownership registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class DocumentRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    application_id: str
    document_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: str = "active"
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    retained_until: datetime


class DocumentRegistry(Protocol):
    def register(self, record: DocumentRecord) -> None: ...

    def owns(
        self, application_id: str, document_ids: tuple[str, ...], *, policy_hash: str
    ) -> bool: ...

    def delete_application(self, application_id: str, policy_hash: str) -> tuple[str, ...]: ...


class InMemoryDocumentRegistry:
    def __init__(self, retention_days: int = 30) -> None:
        self.retention_days = retention_days
        self._records: dict[tuple[str, str], DocumentRecord] = {}
        self._deletion_receipts: dict[str, tuple[str, ...]] = {}
        self._lock = Lock()

    def record(
        self, application_id: str, document_id: str, sha256: str, policy_hash: str
    ) -> DocumentRecord:
        return DocumentRecord(
            application_id=application_id,
            document_id=document_id,
            sha256=sha256,
            policy_hash=policy_hash,
            retained_until=datetime.now(UTC) + timedelta(days=self.retention_days),
        )

    def register(self, record: DocumentRecord) -> None:
        with self._lock:
            key = (record.application_id, record.document_id)
            existing = self._records.get(key)
            if existing is not None and existing.sha256 != record.sha256:
                raise ValueError("document identity conflicts with registered content")
            self._records[key] = record

    def owns(
        self, application_id: str, document_ids: tuple[str, ...], *, policy_hash: str
    ) -> bool:
        if not document_ids:
            return False
        with self._lock:
            return all(
                (record := self._records.get((application_id, document_id))) is not None
                and record.state == "active"
                and record.policy_hash == policy_hash
                for document_id in document_ids
            )

    def delete_application(self, application_id: str, policy_hash: str) -> tuple[str, ...]:
        with self._lock:
            document_ids = tuple(
                document_id
                for (owner, document_id), record in self._records.items()
                if owner == application_id and record.state != "deleted"
            )
            for document_id in document_ids:
                key = (application_id, document_id)
                self._records[key] = self._records[key].model_copy(update={"state": "deleted"})
            if document_ids:
                self._deletion_receipts[application_id] = tuple(sorted(document_ids))
            return self._deletion_receipts.get(application_id, ())


class PostgresSecurityStore:
    """Durable ownership registry and one-time service-token replay ledger."""

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("FinDocIQ database URL is required")
        self.dsn = dsn

    def register(self, record: DocumentRecord) -> None:
        import psycopg

        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO findociq.documents
                    (application_id, document_id, sha256, state, policy_hash, retained_until)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (application_id, document_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    retained_until = GREATEST(findociq.documents.retained_until,
                                              EXCLUDED.retained_until)
                WHERE findociq.documents.sha256 = EXCLUDED.sha256
                """,
                (
                    record.application_id,
                    record.document_id,
                    record.sha256,
                    record.state,
                    record.policy_hash,
                    record.retained_until,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("document identity conflicts with registered content")

    def owns(
        self, application_id: str, document_ids: tuple[str, ...], *, policy_hash: str
    ) -> bool:
        if not document_ids:
            return False
        import psycopg

        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(DISTINCT document_id)
                FROM findociq.documents
                WHERE application_id = %s AND document_id = ANY(%s)
                  AND state = 'active' AND policy_hash = %s
                """,
                (application_id, list(document_ids), policy_hash),
            )
            return cursor.fetchone()[0] == len(set(document_ids))

    def consume(self, jti: str, expires_at: datetime) -> bool:
        import psycopg

        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM findociq.jwt_replay WHERE expires_at <= now()")
            cursor.execute(
                """
                INSERT INTO findociq.jwt_replay (jti, expires_at)
                VALUES (%s, %s) ON CONFLICT (jti) DO NOTHING
                """,
                (jti, expires_at),
            )
            return cursor.rowcount == 1

    def delete_application(self, application_id: str, policy_hash: str) -> tuple[str, ...]:
        import psycopg

        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE findociq.documents SET state = 'deleted', updated_at = now()
                WHERE application_id = %s AND state != 'deleted'
                RETURNING document_id, sha256
                """,
                (application_id,),
            )
            hashes = {row[0]: row[1] for row in cursor.fetchall()}
            cursor.execute(
                """
                INSERT INTO findociq.deletion_receipts
                    (application_id, policy_hash, artifact_hashes)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (application_id) DO NOTHING
                """,
                (application_id, policy_hash, json.dumps(hashes, sort_keys=True)),
            )
            if not hashes:
                cursor.execute(
                    """
                    SELECT artifact_hashes FROM findociq.deletion_receipts
                    WHERE application_id = %s
                    """,
                    (application_id,),
                )
                row = cursor.fetchone()
                if row is not None:
                    stored = row[0]
                    if isinstance(stored, str):
                        stored = json.loads(stored)
                    hashes = dict(stored)
            return tuple(sorted(hashes))
