"""Short-lived, application-scoped service JWT verification."""

from __future__ import annotations

import hmac
import json
from base64 import urlsafe_b64decode
from datetime import UTC, datetime
from hashlib import sha256
from threading import Lock
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from findociq.security.policy import ServiceAuthPolicy


class ServiceClaims(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    sub: str = Field(min_length=1)
    aud: str | tuple[str, ...]
    iss: str
    roles: frozenset[str] = Field(min_length=1)
    application_id: str = Field(min_length=1)
    jti: str = Field(min_length=8)
    iat: int
    exp: int


class ReplayLedger(Protocol):
    def consume(self, jti: str, expires_at: datetime) -> bool: ...


class InMemoryReplayLedger:
    """Process-local development ledger; production composition supplies PostgreSQL."""

    def __init__(self) -> None:
        self._entries: dict[str, datetime] = {}
        self._lock = Lock()

    def consume(self, jti: str, expires_at: datetime) -> bool:
        now = datetime.now(UTC)
        with self._lock:
            self._entries = {key: value for key, value in self._entries.items() if value > now}
            if jti in self._entries:
                return False
            self._entries[jti] = expires_at
            return True


class ServiceJwtVerifier:
    def __init__(
        self,
        secret: str,
        policy: ServiceAuthPolicy,
        replay_ledger: ReplayLedger | None = None,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("service JWT secret must contain at least 32 characters")
        self.secret = secret
        self.policy = policy
        self.replay_ledger = replay_ledger or InMemoryReplayLedger()

    def verify(
        self,
        token: str,
        *,
        required_role: str,
        application_id: str,
        consume: bool = True,
    ) -> ServiceClaims:
        try:
            encoded_header, encoded_payload, encoded_signature = token.split(".")
            header = json.loads(_decode_segment(encoded_header))
            if header.get("alg") != "HS256" or header.get("typ", "JWT") != "JWT":
                raise ValueError("unsupported service token algorithm")
            signed = f"{encoded_header}.{encoded_payload}".encode()
            expected = hmac.new(self.secret.encode(), signed, sha256).digest()
            supplied = urlsafe_b64decode(encoded_signature + "=" * (-len(encoded_signature) % 4))
            if not hmac.compare_digest(expected, supplied):
                raise ValueError("invalid service token signature")
            payload = json.loads(_decode_segment(encoded_payload))
            claims = ServiceClaims.model_validate(payload)
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            raise ValueError("invalid service authorization") from error
        now = int(datetime.now(UTC).timestamp())
        audiences = {claims.aud} if isinstance(claims.aud, str) else set(claims.aud)
        if claims.iss != self.policy.issuer or self.policy.audience not in audiences:
            raise ValueError("invalid service token issuer or audience")
        if claims.iat > now + 30 or claims.exp <= now:
            raise ValueError("service token is expired or not yet valid")
        if claims.application_id != application_id:
            raise PermissionError("service token is scoped to a different application")
        if required_role not in claims.roles:
            raise PermissionError("service role is not authorized for this operation")
        if claims.exp - claims.iat > self.policy.maximum_token_age_seconds:
            raise ValueError("service token lifetime exceeds policy")
        if consume and not self.replay_ledger.consume(
            claims.jti, datetime.fromtimestamp(claims.exp, UTC)
        ):
            raise ValueError("service token replay detected")
        return claims


def _decode_segment(value: str) -> str:
    return urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("utf-8")
