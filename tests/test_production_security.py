import hmac
import json
from base64 import urlsafe_b64encode
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from findociq.security.auth import InMemoryReplayLedger, ServiceJwtVerifier
from findociq.security.dlp import LocalDlpProvider
from findociq.security.policy import ProductionGuardrailPolicy
from findociq.security.registry import InMemoryDocumentRegistry
from findociq.security.storage import EnvelopeEncryptedStore

ROOT = Path(__file__).parents[1]


def policy() -> ProductionGuardrailPolicy:
    return ProductionGuardrailPolicy.from_yaml(ROOT / "configs/guardrails/production.yaml")


def _jwt(secret: str, application_id: str, *, role: str = "document_extract") -> str:
    now = datetime.now(UTC)
    header = _segment({"alg": "HS256", "typ": "JWT"})
    payload = _segment(
        {
            "sub": "fundermatch",
            "iss": "fundermatch",
            "aud": "findociq-api",
            "roles": [role],
            "application_id": application_id,
            "jti": "production-jti-0001",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        }
    )
    signed = f"{header}.{payload}"
    signature = urlsafe_b64encode(
        hmac.new(secret.encode(), signed.encode(), sha256).digest()
    ).rstrip(b"=").decode()
    return f"{signed}.{signature}"


def _segment(value: dict[str, object]) -> str:
    return urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(
        b"="
    ).decode()


def test_shared_policy_is_stable_and_dlp_never_returns_matched_values() -> None:
    current = policy()
    assert len(current.policy_hash) == 64
    result = LocalDlpProvider(current.dlp, current.policy_hash).inspect(
        "PAN ABCDE1234F email person@example.com Address: 12 MG Road, Bengaluru"
    )
    assert "ABCDE1234F" not in result.redacted_text
    assert "person@example.com" not in result.redacted_text
    assert result.receipt.counts == {"pan": 1, "email": 1, "address": 1}
    assert "ABCDE1234F" not in result.receipt.model_dump_json()


def test_service_jwt_is_application_scoped_role_limited_and_one_time() -> None:
    secret = "s" * 32
    verifier = ServiceJwtVerifier(secret, policy().service_auth, InMemoryReplayLedger())
    token = _jwt(secret, "APP-SEC-1")
    claims = verifier.verify(
        token, required_role="document_extract", application_id="APP-SEC-1"
    )
    assert claims.application_id == "APP-SEC-1"
    with pytest.raises(ValueError, match="replay"):
        verifier.verify(token, required_role="document_extract", application_id="APP-SEC-1")
    with pytest.raises(PermissionError, match="different application"):
        verifier.verify(
            _jwt(secret, "APP-SEC-1"),
            required_role="document_extract",
            application_id="APP-SEC-2",
        )


def test_registry_rejects_cross_application_document_access() -> None:
    current = policy()
    registry = InMemoryDocumentRegistry()
    record = registry.record("APP-SEC-1", "doc-1", "a" * 64, current.policy_hash)
    registry.register(record)
    assert registry.owns("APP-SEC-1", ("doc-1",))
    assert not registry.owns("APP-SEC-2", ("doc-1",))
    assert not registry.owns("APP-SEC-1", ())


def test_envelope_store_keeps_plaintext_out_of_persistent_file(tmp_path: Path) -> None:
    store = EnvelopeEncryptedStore(tmp_path, b"k" * 32)
    plaintext = b"%PDF-1.7 borrower PAN ABCDE1234F"
    digest = sha256(plaintext).hexdigest()
    path = store.store(
        application_id="APP-SEC-1",
        document_id=digest,
        sha256=digest,
        content=plaintext,
    )
    assert plaintext not in path.read_bytes()
    with store.materialize(path, "borrower.pdf") as materialized:
        assert materialized.read_bytes() == plaintext
    assert not materialized.exists()


def test_envelope_key_rotation_rewraps_without_rewriting_pdf_ciphertext(
    tmp_path: Path,
) -> None:
    old_key = b"o" * 32
    new_key = b"n" * 32
    content = b"%PDF-1.7\nconfidential"
    store = EnvelopeEncryptedStore(tmp_path, old_key, "v1")
    target = store.store(
        application_id="APP-ROTATE",
        document_id="doc-1",
        sha256=sha256(content).hexdigest(),
        content=content,
    )
    before = json.loads(target.read_text(encoding="utf-8"))

    assert store.rewrap_all(new_key, "v2") == 1
    after = json.loads(target.read_text(encoding="utf-8"))
    assert after["ciphertext"] == before["ciphertext"]
    assert after["wrapped_key"] != before["wrapped_key"]
    with EnvelopeEncryptedStore(tmp_path, new_key, "v2").materialize(
        target, "document.pdf"
    ) as materialized:
        assert materialized.read_bytes() == content


def test_document_retention_delete_is_idempotent() -> None:
    registry = InMemoryDocumentRegistry()
    record = registry.record("APP-DELETE", "doc-1", "d" * 64, "a" * 64)
    registry.register(record)
    assert registry.delete_application("APP-DELETE", "a" * 64) == ("doc-1",)
    assert registry.delete_application("APP-DELETE", "a" * 64) == ("doc-1",)
