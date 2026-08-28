"""Application-scoped storage for inputs and worker artifacts.

This store is intentionally separate from LangGraph checkpoints.  It may contain
borrower content, while checkpoints contain only compact references and hashes.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from fundermatch.intake import IntakeMetadata
from fundermatch.orchestration.schema import InputReference

_Model = TypeVar("_Model", bound=BaseModel)


class StagedFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StagedApplication(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: IntakeMetadata
    files: tuple[StagedFile, ...] = Field(min_length=1)


class ApplicationWorkspace:
    """Durable, application-isolated worker artifact repository."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def stage(
        self,
        metadata: IntakeMetadata,
        files: tuple[tuple[str, bytes], ...],
        *,
        max_file_bytes: int,
        max_batch_bytes: int,
    ) -> tuple[InputReference, ...]:
        if not files:
            raise ValueError("at least one borrower PDF is required")
        if sum(len(content) for _, content in files) > max_batch_bytes:
            raise ValueError("PDF batch exceeds the configured aggregate limit")
        folder = self._folder(metadata.application_id)
        documents = folder / "documents"
        documents.mkdir(parents=True, exist_ok=True)
        staged: list[StagedFile] = []
        for filename, content in files:
            safe_name = Path(filename).name
            if safe_name != filename or not safe_name.lower().endswith(".pdf"):
                raise ValueError("all uploaded files must be safely named PDFs")
            if not content.startswith(b"%PDF-"):
                raise ValueError(f"{safe_name} is not a PDF")
            if len(content) > max_file_bytes:
                raise ValueError(f"{safe_name} exceeds the configured PDF limit")
            digest = sha256(content).hexdigest()
            target = documents / safe_name
            if target.exists() and sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"staged file collision for {safe_name}")
            if not target.exists():
                target.write_bytes(content)
            staged.append(StagedFile(filename=safe_name, sha256=digest))
        request = StagedApplication(metadata=metadata, files=tuple(staged))
        self.save(metadata.application_id, "request", request)
        return tuple(
            InputReference(
                reference_type="staged_pdf",
                reference_id=item.filename,
                sha256=item.sha256,
            )
            for item in staged
        )

    def request(self, application_id: str) -> StagedApplication:
        return self.load(application_id, "request", StagedApplication)

    def read_pdf(self, application_id: str, filename: str, expected_sha256: str) -> bytes:
        safe_name = Path(filename).name
        if safe_name != filename:
            raise ValueError("unsafe staged filename")
        payload = (self._folder(application_id) / "documents" / safe_name).read_bytes()
        if sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("staged PDF hash mismatch")
        return payload

    def exists(self, application_id: str, name: str) -> bool:
        return self._artifact_path(application_id, name).exists()

    def save(self, application_id: str, name: str, value: BaseModel) -> str:
        target = self._artifact_path(application_id, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = value.model_dump_json(indent=2)
        digest = sha256(payload.encode("utf-8")).hexdigest()
        temporary = target.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
        return digest

    def load(self, application_id: str, name: str, model: type[_Model]) -> _Model:
        return model.model_validate_json(
            self._artifact_path(application_id, name).read_text(encoding="utf-8")
        )

    def artifact_hash(self, application_id: str, name: str) -> str:
        payload = self._artifact_path(application_id, name).read_bytes()
        return sha256(payload).hexdigest()

    def _artifact_path(self, application_id: str, name: str) -> Path:
        if not name or any(part in {"", ".", ".."} for part in Path(name).parts):
            raise ValueError("unsafe artifact name")
        return self._folder(application_id) / "agent" / f"{name}.json"

    def _folder(self, application_id: str) -> Path:
        if not application_id or Path(application_id).name != application_id:
            raise ValueError("unsafe application identifier")
        folder = (self.root / application_id).resolve()
        if self.root not in folder.parents:
            raise ValueError("application storage boundary violation")
        return folder
