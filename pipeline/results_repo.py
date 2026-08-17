import os
import threading
from typing import Dict, Optional


class ResultsRepository:
    def init_db(self) -> None:
        raise NotImplementedError

    def save(self, filename: str, results: dict) -> None:
        raise NotImplementedError

    def get(self, filename: str) -> Optional[dict]:
        raise NotImplementedError


class InMemoryResultsRepository(ResultsRepository):
    def __init__(self):
        self._rows: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def init_db(self) -> None:
        pass

    def save(self, filename: str, results: dict) -> None:
        with self._lock:
            self._rows[filename] = results

    def get(self, filename: str) -> Optional[dict]:
        with self._lock:
            return self._rows.get(filename)


class PostgresResultsRepository(ResultsRepository):
    def __init__(self, url: str = "postgresql://svar:svar@localhost:5432/svar"):
        import psycopg
        self._conn = psycopg.connect(url, autocommit=True)

    def init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                results JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    def save(self, filename: str, results: dict) -> None:
        import json
        self._conn.execute(
            """
            INSERT INTO calls (filename, results) VALUES (%s, %s)
            ON CONFLICT (filename) DO UPDATE SET results = EXCLUDED.results
            """,
            (filename, json.dumps(results)),
        )

    def get(self, filename: str) -> Optional[dict]:
        import json
        row = self._conn.execute(
            "SELECT results FROM calls WHERE filename = %s", (filename,)
        ).fetchone()
        if not row:
            return None
        val = row[0]
        return json.loads(val) if isinstance(val, str) else val