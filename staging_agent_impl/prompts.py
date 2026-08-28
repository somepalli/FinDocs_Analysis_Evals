"""Single typed boundary for repository-owned prompt files."""

from pathlib import Path


def load_prompt(name: str, root: Path = Path("prompts")) -> str:
    if not name or Path(name).name != name:
        raise ValueError("unsafe prompt name")
    prompt = (root / f"{name}.txt").read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt {name!r} is empty")
    return prompt
