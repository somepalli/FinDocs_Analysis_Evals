"""Local deterministic DLP classification and redaction."""

from __future__ import annotations

import re
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field

from findociq.security.policy import DlpPolicy

_PATTERNS: dict[str, re.Pattern[str]] = {
    "aadhaar": re.compile(r"(?<!\d)(?:\d[ -]?){11}\d(?!\d)"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE),
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)"),
    "bank_account": re.compile(r"\b\d{9,18}\b"),
    "address": re.compile(
        r"\b(?:registered\s+office|residential\s+address|address)\s*[:\-]\s*[^\n]{5,250}",
        re.IGNORECASE,
    ),
    "credential": re.compile(
        r"\b(?:password|passwd|secret|api[_ -]?key|bearer)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    "tax_identifier": re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b", re.IGNORECASE),
}


class DlpReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counts: dict[str, int]
    instruction_risk: bool
    disposition: str


class DlpResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    redacted_text: str
    receipt: DlpReceipt


class LocalDlpProvider:
    def __init__(self, policy: DlpPolicy, policy_hash: str) -> None:
        self.policy = policy
        self.policy_hash = policy_hash
        unknown = set(policy.protected_kinds) - _PATTERNS.keys()
        if unknown:
            raise ValueError(f"unsupported DLP kinds: {sorted(unknown)}")
        self._instruction_patterns = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in policy.instruction_patterns
        )

    def inspect(self, text: str) -> DlpResult:
        redacted = text
        counts: dict[str, int] = {}
        for kind in self.policy.protected_kinds:
            pattern = _PATTERNS[kind]
            redacted, count = pattern.subn(self.policy.replacement.format(kind=kind), redacted)
            if count:
                counts[kind] = count
        instruction_risk = any(pattern.search(text) for pattern in self._instruction_patterns)
        return DlpResult(
            redacted_text=redacted,
            receipt=DlpReceipt(
                policy_hash=self.policy_hash,
                artifact_sha256=sha256(text.encode()).hexdigest(),
                counts=counts,
                instruction_risk=instruction_risk,
                disposition="needs_attention" if instruction_risk else "accepted",
            ),
        )
