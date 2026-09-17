from datetime import UTC, datetime

import pytest

from findociq.security.document_sets import DocumentSetStore, freeze
from findociq.security.registry import DocumentRecord, InMemoryDocumentRegistry


def registry(count):
    result = InMemoryDocumentRegistry()
    for n in range(count):
        result.register(
            DocumentRecord(
                application_id="APP-A",
                document_id=str(n),
                sha256=f"{n:064x}",
                policy_hash="a" * 64,
                retained_until=datetime.now(UTC),
            )
        )
    return result


@pytest.mark.parametrize("count", [5, 20, 21, 37])
def test_immutable_large_sets_survive_reopen(tmp_path, count):
    r = registry(count)
    record = freeze("APP-A", tuple(str(n) for n in range(count)), "a" * 64)
    store = DocumentSetStore(path=tmp_path / "sets.sqlite")
    assert store.put(record, r.owns) == store.put(record, r.owns)
    reopened = DocumentSetStore(path=tmp_path / "sets.sqlite")
    assert reopened.get(record.set_id, "APP-A", "a" * 64, r.owns) == record
    assert len(record.document_ids) == count
    with pytest.raises(PermissionError):
        reopened.get(record.set_id, "APP-B", "a" * 64, r.owns)
    r.delete_application("APP-A", "a" * 64)
    with pytest.raises(PermissionError):
        reopened.get(record.set_id, "APP-A", "a" * 64, r.owns)


def test_canonical_hash_deduplicates_and_changes_with_membership():
    assert freeze("APP-A", ("2", "1", "1"), "a" * 64) == freeze("APP-A", ("1", "2"), "a" * 64)
    assert freeze("APP-A", ("1",), "a" * 64).set_id != freeze("APP-A", ("1", "2"), "a" * 64).set_id
