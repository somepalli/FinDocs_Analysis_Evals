"""Company age from an explicit incorporation date, not commencement inference."""

from datetime import date, datetime


def incorporation_date(value: str) -> date:
    for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            pass
    raise ValueError("invalid incorporation date")


def completed_years(incorporated: date, assessed: date) -> int:
    if incorporated > assessed:
        raise ValueError("incorporation date is after assessment date")
    return (
        assessed.year
        - incorporated.year
        - ((assessed.month, assessed.day) < (incorporated.month, incorporated.day))
    )
