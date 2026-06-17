from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from agent_memory.schemas import Message, datetime_from_iso, datetime_to_iso


class SQLiteRecallStore:
    """SQLite-backed append-only store for raw conversation messages."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def add_message(self, message: Message) -> Message:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    id, user_id, session_id, role, content, created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.user_id,
                    message.session_id,
                    message.role,
                    message.content,
                    datetime_to_iso(message.created_at),
                    json.dumps(message.metadata, ensure_ascii=False),
                ),
            )
        return message

    def add_messages(self, messages: Iterable[Message]) -> list[Message]:
        saved = list(messages)
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO messages (
                    id, user_id, session_id, role, content, created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        message.id,
                        message.user_id,
                        message.session_id,
                        message.role,
                        message.content,
                        datetime_to_iso(message.created_at),
                        json.dumps(message.metadata, ensure_ascii=False),
                    )
                    for message in saved
                ],
            )
        return saved

    def get_message(self, message_id: str) -> Message | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        return self._row_to_message(row) if row else None

    def list_messages(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Message]:
        params: list[object] = [user_id]
        where = "user_id = ?"
        if session_id is not None:
            where += " AND session_id = ?"
            params.append(session_id)

        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM messages
                WHERE {where}
                ORDER BY created_at ASC, rowid ASC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def search_messages(self, user_id: str, query: str, limit: int = 20) -> list[Message]:
        pattern = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE user_id = ? AND content LIKE ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (user_id, pattern, limit),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

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
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_user_session_created
                ON messages (user_id, session_id, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_user_created
                ON messages (user_id, created_at)
                """
            )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=datetime_from_iso(row["created_at"]),
            metadata=json.loads(row["metadata"]),
        )
