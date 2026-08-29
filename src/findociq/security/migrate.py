"""Apply FinDocIQ production-security PostgreSQL migrations."""

from pathlib import Path

import psycopg

from findociq.security.secrets import read_secret


def main() -> None:
    root = Path(__file__).parents[3]
    with psycopg.connect(read_secret("FINDOCIQ_DATABASE_URL")) as connection:
        for migration in sorted((root / "migrations").glob("*.sql")):
            connection.execute(migration.read_text(encoding="utf-8"))
            print(f"applied {migration.name}")


if __name__ == "__main__":
    main()
