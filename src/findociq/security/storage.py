"""AES-256-GCM envelope-encrypted document storage."""

from __future__ import annotations

import json
import os
import shutil
from base64 import b64decode, b64encode
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EnvelopeEncryptedStore:
    """Encrypt each object with a DEK and wrap that DEK with a versioned master key."""

    def __init__(self, root: Path, master_key: bytes, key_version: str = "v1") -> None:
        if len(master_key) != 32:
            raise ValueError("document master key must be exactly 32 bytes")
        self.root = root.resolve()
        self.master_key = master_key
        self.key_version = key_version

    @classmethod
    def from_secret_file(
        cls, root: Path, secret_file: str | Path, key_version: str
    ) -> EnvelopeEncryptedStore:
        encoded = Path(secret_file).read_text(encoding="utf-8").strip()
        return cls(root, b64decode(encoded, validate=True), key_version)

    def store(
        self,
        *,
        application_id: str,
        document_id: str,
        sha256: str,
        content: bytes,
        state: str = "active",
    ) -> Path:
        associated = self._associated(application_id, document_id, sha256, state)
        dek = AESGCM.generate_key(bit_length=256)
        data_nonce = os.urandom(12)
        key_nonce = os.urandom(12)
        encrypted = AESGCM(dek).encrypt(data_nonce, content, associated)
        wrapped = AESGCM(self.master_key).encrypt(key_nonce, dek, associated)
        payload = {
            "version": 1,
            "key_version": self.key_version,
            "application_id": application_id,
            "document_id": document_id,
            "sha256": sha256,
            "state": state,
            "data_nonce": b64encode(data_nonce).decode(),
            "key_nonce": b64encode(key_nonce).decode(),
            "ciphertext": b64encode(encrypted).decode(),
            "wrapped_key": b64encode(wrapped).decode(),
        }
        folder = self.root / state / application_id
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{document_id}.enc.json"
        temporary = folder / f".{document_id}.enc.upload"
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        return target

    @contextmanager
    def materialize(self, encrypted_path: Path, filename: str) -> Iterator[Path]:
        payload = json.loads(encrypted_path.read_text(encoding="utf-8"))
        associated = self._associated(
            payload["application_id"], payload["document_id"], payload["sha256"], payload["state"]
        )
        dek = AESGCM(self.master_key).decrypt(
            b64decode(payload["key_nonce"]), b64decode(payload["wrapped_key"]), associated
        )
        plaintext = AESGCM(dek).decrypt(
            b64decode(payload["data_nonce"]), b64decode(payload["ciphertext"]), associated
        )
        with TemporaryDirectory(prefix="findociq-plaintext-") as folder:
            path = Path(folder) / filename
            path.write_bytes(plaintext)
            try:
                yield path
            finally:
                if path.exists():
                    path.write_bytes(b"\0" * path.stat().st_size)
                    path.unlink(missing_ok=True)

    def delete_application(self, application_id: str) -> None:
        for state in ("active", "quarantined"):
            folder = (self.root / state / application_id).resolve()
            if self.root not in folder.parents:
                raise ValueError("encrypted storage deletion escaped configured root")
            if folder.exists():
                shutil.rmtree(folder)

    def rewrap_all(self, new_master_key: bytes, new_key_version: str) -> int:
        """Rewrap object DEKs without decrypting or rewriting PDF ciphertext."""

        if len(new_master_key) != 32:
            raise ValueError("new document master key must be exactly 32 bytes")
        changed = 0
        for state in ("active", "quarantined"):
            folder = self.root / state
            if not folder.exists():
                continue
            for target in folder.rglob("*.enc.json"):
                resolved = target.resolve()
                if self.root not in resolved.parents:
                    raise ValueError("encrypted object escaped configured root")
                payload = json.loads(resolved.read_text(encoding="utf-8"))
                associated = self._associated(
                    payload["application_id"],
                    payload["document_id"],
                    payload["sha256"],
                    payload["state"],
                )
                dek = AESGCM(self.master_key).decrypt(
                    b64decode(payload["key_nonce"]),
                    b64decode(payload["wrapped_key"]),
                    associated,
                )
                key_nonce = os.urandom(12)
                payload["key_version"] = new_key_version
                payload["key_nonce"] = b64encode(key_nonce).decode()
                payload["wrapped_key"] = b64encode(
                    AESGCM(new_master_key).encrypt(key_nonce, dek, associated)
                ).decode()
                temporary = resolved.with_suffix(".rewrap")
                temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                temporary.replace(resolved)
                changed += 1
        self.master_key = new_master_key
        self.key_version = new_key_version
        return changed

    @staticmethod
    def _associated(application_id: str, document_id: str, sha256: str, state: str) -> bytes:
        return f"{application_id}|{document_id}|{sha256}|{state}".encode()
