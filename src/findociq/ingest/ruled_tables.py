"""Recover horizontally ruled digital tables using explicit dated column anchors."""

import re

import fitz


def activity_tables(page):
    """Recover activity rows below a detected header, bounded by the next section.

    Some PDFs have vertical borders only in the header. The normal detector then
    returns a header-only table whose lower edge slightly overlaps the body text.
    Use those explicit column boundaries and actual horizontal body rules; never
    synthesize a row from prose outside the section.
    """
    drawings = page.get_drawings()
    results = []
    for table in page.find_tables().tables:
        raw = table.extract()
        if len(raw) != 1 or len(raw[0]) != 5:
            continue
        header = [(c or "").replace("\n", " ").strip() for c in raw[0]]
        if "Description of Main Activity Group" not in header or "% of Turnover" not in header:
            continue
        box = fitz.Rect(table.bbox)
        sections = [
            fitz.Rect(d["rect"])
            for d in drawings
            if d.get("fill") is not None
            and fitz.Rect(d["rect"]).height > 8
            and fitz.Rect(d["rect"]).width >= box.width * 0.9
            and fitz.Rect(d["rect"]).y0 > box.y1
        ]
        if not sections:
            continue
        end = min(r.y0 for r in sections)
        bottoms = sorted(
            {
                float(item[1].y)
                for d in drawings
                for item in d["items"]
                if item[0] == "l"
                and abs(item[1].y - item[2].y) < 0.5
                and abs(item[2].x - item[1].x) >= box.width * 0.9
                and box.y1 + 2 < item[1].y < end
            }
        )
        cells = table.rows[0].cells
        if not bottoms or any(c is None for c in cells):
            continue
        rows = []
        top = box.y1
        for bottom in bottoms:
            row = [
                " ".join(
                    w[4]
                    for w in sorted(page.get_text("words"), key=lambda w: (round(w[1], 1), w[0]))
                    if c[0] <= (w[0] + w[2]) / 2 < c[2] and top < (w[1] + w[3]) / 2 < bottom
                )
                for c in cells
            ]
            if not (
                all(row)
                and re.fullmatch(r"[A-Z]", row[0])
                and re.fullmatch(r"\d+(?:\.\d+)?", row[2])
                and re.fullmatch(r"\d+(?:\.\d+)?", row[4])
            ):
                rows = []
                break
            rows.append(row)
            top = bottom
        if rows:
            markdown = "\n".join("| " + " | ".join(r) + " |" for r in [header, ["---"] * 5, *rows])
            results.append((fitz.Rect(box.x0, box.y0, box.x1, bottoms[-1]), markdown))
    return results


def dated_tables(page):
    words = page.get_text("words", sort=True)
    lines = []
    for word in sorted(words, key=lambda w: (round(w[1], 1), w[0])):
        line = next((line for line in lines if abs(line[0][1] - word[1]) < 2), None)
        if line is None:
            lines.append([word])
        else:
            line.append(word)
    headers = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        dates = []
        for i in range(len(line) - 2):
            if (
                re.fullmatch(r"\d{1,2}", line[i][4])
                and re.fullmatch(
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),?", line[i + 1][4]
                )
                and re.fullmatch(r"20\d{2}", line[i + 2][4])
            ):
                dates.append((line[i][0], line[i + 2][2]))
        if 2 <= len(dates) <= 10 and line[0][0] < dates[0][0] - 30:
            headers.append((line, dates))
    results = []
    for index, (line, dates) in enumerate(headers):
        top = min(w[1] for w in line) - 8
        bottom = (
            min(w[1] for w in headers[index + 1][0]) - 8
            if index + 1 < len(headers)
            else page.rect.height - 80
        )
        left = min(w[0] for w in line)
        right = max(w[2] for w in line) + 8
        verticals = [left, dates[0][0] - 5]
        verticals.extend((dates[i - 1][1] + dates[i][0]) / 2 for i in range(1, len(dates)))
        verticals.append(right)
        found = page.find_tables(
            clip=fitz.Rect(left - 1, top, right, bottom),
            vertical_strategy="explicit",
            vertical_lines=verticals,
            horizontal_strategy="lines",
        ).tables
        for table in found:
            rows = [
                [(cell or "").replace("\n", " ").strip() for cell in row] for row in table.extract()
            ]
            if not rows or len(rows[0]) != len(dates) + 1:
                continue
            dated_header = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if all(re.search(r"\b20\d{2}\b", c) for c in row[1:])
                ),
                None,
            )
            if dated_header is None:
                continue
            rows = rows[dated_header:]
            numeric = sum(
                bool(row[0]) and all(re.fullmatch(r"[-+()\d,.% ]+", c) for c in row[1:])
                for row in rows[1:]
            )
            if numeric < 3:
                continue
            markdown = "\n".join(
                "| " + " | ".join(row) + " |"
                for row in [rows[0], ["---"] * len(rows[0]), *rows[1:]]
            )
            results.append((fitz.Rect(table.bbox), markdown))
    return results
