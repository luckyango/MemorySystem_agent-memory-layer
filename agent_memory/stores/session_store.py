from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agent_memory.schemas import SessionState, datetime_from_iso, datetime_to_iso, utc_now


class SQLiteSessionStore:
    """SQLite store for short-term session state."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def get_session(self, user_id: str, session_id: str) -> SessionState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
        return self._row_to_session(row) if row else None

    def upsert_session(self, state: SessionState) -> SessionState:
        state.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    user_id, session_id, active_project_id, current_task,
                    temporary_constraints, metadata, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    active_project_id = excluded.active_project_id,
                    current_task = excluded.current_task,
                    temporary_constraints = excluded.temporary_constraints,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                self._session_to_row(state),
            )
        return state

    def update_session(
        self,
        *,
        user_id: str,
        session_id: str,
        active_project_id: str | None = None,
        current_task: str | None = None,
        temporary_constraints: list[str] | None = None,
        metadata: dict | None = None,
    ) -> SessionState:
        existing = self.get_session(user_id, session_id) or SessionState(
            user_id=user_id,
            session_id=session_id,
        )
        if active_project_id is not None:
            existing.active_project_id = active_project_id
        if current_task is not None:
            existing.current_task = current_task
        if temporary_constraints is not None:
            existing.temporary_constraints = temporary_constraints
        if metadata is not None:
            existing.metadata = metadata
        return self.upsert_session(existing)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    active_project_id TEXT,
                    current_task TEXT,
                    temporary_constraints TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions (user_id, updated_at)
                """
            )

    @staticmethod
    def _session_to_row(state: SessionState) -> tuple[object, ...]:
        return (
            state.user_id,
            state.session_id,
            state.active_project_id,
            state.current_task,
            json.dumps(state.temporary_constraints, ensure_ascii=False),
            json.dumps(state.metadata, ensure_ascii=False),
            datetime_to_iso(state.updated_at),
        )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> SessionState:
        return SessionState(
            user_id=row["user_id"],
            session_id=row["session_id"],
            active_project_id=row["active_project_id"],
            current_task=row["current_task"],
            temporary_constraints=json.loads(row["temporary_constraints"]),
            metadata=json.loads(row["metadata"]),
            updated_at=datetime_from_iso(row["updated_at"]),
        )
