"""NIC hierarchy from explicit Udyam tables; never classify from filenames."""

import re

from pydantic import BaseModel, ConfigDict


class NicActivity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    division: str
    class_code: str
    subclass: str
    division_description: str
    class_description: str
    subclass_description: str
    activity: str


def activities(text: str) -> tuple[NicActivity, ...]:
    headers = None
    found = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        normalized = [re.sub(r"\s+", " ", c).lower() for c in cells]
        if all(
            label in normalized
            for label in ("nic 2 digit", "nic 4 digit", "nic 5 digit", "activity")
        ):
            headers = normalized
            continue
        if (
            headers is None
            or len(cells) != len(headers)
            or all(re.fullmatch(r"[-: ]*", c) for c in cells)
        ):
            continue
        values = [
            cells[headers.index(label)] for label in ("nic 2 digit", "nic 4 digit", "nic 5 digit")
        ]
        parsed = [
            re.fullmatch(r"(\d{" + str(n) + r"})\s*[-–:]\s*(\S.*)", value)
            for n, value in zip((2, 4, 5), values, strict=True)
        ]
        if not all(parsed):
            raise ValueError("NIC row is incomplete")
        division, group, subclass = (p.group(1) for p in parsed)
        if not group.startswith(division) or not subclass.startswith(group):
            raise ValueError("NIC hierarchy conflict")
        item = NicActivity(
            division=division,
            class_code=group,
            subclass=subclass,
            division_description=parsed[0].group(2),
            class_description=parsed[1].group(2),
            subclass_description=parsed[2].group(2),
            activity=cells[headers.index("activity")],
        )
        if item not in found:
            found.append(item)
    return tuple(found)
