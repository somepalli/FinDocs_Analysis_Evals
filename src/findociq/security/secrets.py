"""Read production credentials from Docker secret files."""

import os
from pathlib import Path


def read_secret(name: str) -> str:
    secret_file = os.getenv(f"{name}_FILE")
    if secret_file:
        return Path(secret_file).read_text(encoding="utf-8").strip()
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"required secret is not configured: {name}")
