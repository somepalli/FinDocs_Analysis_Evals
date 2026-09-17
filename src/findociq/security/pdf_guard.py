"""Fail-closed PDF active-content and ClamAV scanning boundary."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path

import fitz


class UnsafeDocumentError(ValueError):
    """Document is unsafe or cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class ScanReceipt:
    scanner: str
    scanner_version: str
    status: str
    reason_code: str | None = None


class PdfSafetyScanner:
    def __init__(self, host: str = "clamav", port: int = 3310, timeout_seconds: float = 15) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    def scan(self, path: Path) -> ScanReceipt:
        self._reject_active_content(path)
        version = self._version()
        try:
            with (
                socket.create_connection(
                    (self.host, self.port), timeout=self.timeout_seconds
                ) as connection,
                path.open("rb") as source,
            ):
                connection.sendall(b"zINSTREAM\0")
                while block := source.read(1024 * 1024):
                    connection.sendall(len(block).to_bytes(4, "big") + block)
                connection.sendall((0).to_bytes(4, "big"))
                response = connection.recv(4096).decode(errors="replace")
        except OSError as error:
            raise UnsafeDocumentError("malware_scanner_unavailable") from error
        if "FOUND" in response:
            raise UnsafeDocumentError("malware_detected")
        if "OK" not in response:
            raise UnsafeDocumentError("malware_scan_inconclusive")
        return ScanReceipt(scanner="clamav", scanner_version=version, status="clean")

    def _version(self) -> str:
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout_seconds
            ) as connection:
                connection.sendall(b"zVERSION\0")
                response = connection.recv(1024).decode(errors="replace").strip("\0\r\n ")
        except OSError as error:
            raise UnsafeDocumentError("malware_scanner_unavailable") from error
        if not response:
            raise UnsafeDocumentError("malware_scanner_version_unavailable")
        return response[:200]

    @staticmethod
    def _reject_active_content(path: Path) -> None:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            from findociq.ingest.image_input import validate_image

            try:
                validate_image(path.read_bytes(), path.suffix)
            except ValueError as error:
                raise UnsafeDocumentError(str(error)) from error
            return
        try:
            document = fitz.open(path)
        except Exception as error:
            raise UnsafeDocumentError("malformed_pdf") from error
        try:
            if document.needs_pass:
                raise UnsafeDocumentError("encrypted_pdf")
            if document.embfile_count() > 0:
                raise UnsafeDocumentError("embedded_file")
            raw = path.read_bytes()
            forbidden = (
                b"/JavaScript",
                b"/JS",
                b"/Launch",
                b"/OpenAction",
                b"/RichMedia",
                b"/URI",
                b"/GoToR",
                b"/SubmitForm",
                b"/ImportData",
            )
            if any(marker in raw for marker in forbidden):
                raise UnsafeDocumentError("active_pdf_content")
        finally:
            document.close()
