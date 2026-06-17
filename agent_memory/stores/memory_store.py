from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agent_memory.schemas import MemoryItem, datetime_from_iso, datetime_to_iso, utc_now


class SQLiteMemoryStore:
    """SQLite store for structured long-term memories."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def add_memory(self, memory: MemoryItem) -> MemoryItem:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, scope_type, scope_id, category, content,
                    confidence, importance, source_message_ids, entities,
                    metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._memory_to_row(memory),
            )
        return memory

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(
        self,
        user_id: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        params: list[object] = [user_id]
        where = "user_id = ?"
        if scope_type is not None:
            where += " AND scope_type = ?"
            params.append(scope_type)
        if scope_id is not None:
            where += " AND scope_id = ?"
            params.append(scope_id)
        if category is not None:
            where += " AND category = ?"
            params.append(category)

        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE {where}
                ORDER BY importance DESC, updated_at DESC, rowid DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_memories(self, user_id: str, query: str, limit: int = 20) -> list[MemoryItem]:
        pattern = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND content LIKE ?
                ORDER BY importance DESC, updated_at DESC, rowid DESC
                LIMIT ?
                """,
                (user_id, pattern, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        metadata: dict | None = None,
    ) -> MemoryItem | None:
        existing = self.get_memory(memory_id)
        if existing is None:
            return None

        if content is not None:
            existing.content = content
        if confidence is not None:
            existing.confidence = confidence
        if importance is not None:
            existing.importance = importance
        if metadata is not None:
            existing.metadata = metadata
        existing.updated_at = utc_now()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET content = ?, confidence = ?, importance = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    existing.content,
                    existing.confidence,
                    existing.importance,
                    json.dumps(existing.metadata, ensure_ascii=False),
                    datetime_to_iso(existing.updated_at),
                    memory_id,
                ),
            )
        return existing

    def delete_memory(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

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
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    source_message_ids TEXT NOT NULL DEFAULT '[]',
                    entities TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_scope
                ON memories (user_id, scope_type, scope_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_category
                ON memories (user_id, category)
                """
            )

    @staticmethod
    def _memory_to_row(memory: MemoryItem) -> tuple[object, ...]:
        return (
            memory.id,
            memory.user_id,
            memory.scope_type,
            memory.scope_id,
            memory.category,
            memory.content,
            memory.confidence,
            memory.importance,
            json.dumps(memory.source_message_ids, ensure_ascii=False),
            json.dumps(memory.entities, ensure_ascii=False),
            json.dumps(memory.metadata, ensure_ascii=False),
            datetime_to_iso(memory.created_at),
            datetime_to_iso(memory.updated_at),
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            category=row["category"],
            content=row["content"],
            confidence=row["confidence"],
            importance=row["importance"],
            source_message_ids=json.loads(row["source_message_ids"]),
            entities=json.loads(row["entities"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime_from_iso(row["created_at"]),
            updated_at=datetime_from_iso(row["updated_at"]),
        )
