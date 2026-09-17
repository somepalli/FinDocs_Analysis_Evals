"""Durable document ownership for the explicitly local development profile."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from findociq.security.registry import DocumentRecord


class LocalDocumentRegistry:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS document_owners "
            "(application_id TEXT, document_id TEXT, payload TEXT, "
            "PRIMARY KEY(application_id, document_id))"
        )
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def register(self, record: DocumentRecord):
        with self._connection() as connection:
            old = connection.execute(
                "SELECT payload FROM document_owners WHERE application_id=? AND document_id=?",
                (record.application_id, record.document_id),
            ).fetchone()
            if old and DocumentRecord.model_validate_json(old[0]).sha256 != record.sha256:
                raise ValueError("document_content_conflict")
            connection.execute(
                "INSERT INTO document_owners VALUES (?, ?, ?) "
                "ON CONFLICT(application_id,document_id) DO UPDATE SET payload=excluded.payload",
                (record.application_id, record.document_id, record.model_dump_json()),
            )

    def owns(self, application_id, document_ids, *, policy_hash):
        if not document_ids:
            return False
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM document_owners WHERE application_id=?", (application_id,)
            ).fetchall()
        records = [DocumentRecord.model_validate_json(r[0]) for r in rows]
        return set(document_ids) <= {
            r.document_id for r in records if r.state == "active" and r.policy_hash == policy_hash
        }

    def delete_application(self, application_id, policy_hash):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM document_owners WHERE application_id=?", (application_id,)
            ).fetchall()
            for row in rows:
                record = DocumentRecord.model_validate_json(row[0]).model_copy(
                    update={"state": "deleted"}
                )
                connection.execute(
                    "UPDATE document_owners SET payload=? WHERE application_id=? AND document_id=?",
                    (record.model_dump_json(), application_id, record.document_id),
                )
        return tuple(sorted(DocumentRecord.model_validate_json(r[0]).document_id for r in rows))
