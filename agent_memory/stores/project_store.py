from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agent_memory.schemas import Project, datetime_from_iso, datetime_to_iso, utc_now


class SQLiteProjectStore:
    """SQLite-backed registry for user projects and aliases."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def add_project(self, project: Project) -> Project:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, user_id, name, aliases, description, status,
                    created_at, updated_at, last_mentioned_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._project_to_row(project),
            )
        return project

    def get_project(self, project_id: str) -> Project | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self, user_id: str, status: str | None = None) -> list[Project]:
        params: list[object] = [user_id]
        where = "user_id = ?"
        if status is not None:
            where += " AND status = ?"
            params.append(status)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM projects
                WHERE {where}
                ORDER BY updated_at DESC, rowid DESC
                """,
                params,
            ).fetchall()
        return [self._row_to_project(row) for row in rows]

    def find_project(self, user_id: str, text: str) -> Project | None:
        normalized = text.casefold()
        for project in self.list_projects(user_id):
            names = [project.name, *project.aliases]
            if any(name.casefold() in normalized for name in names if name):
                return project
        return None

    def touch_project(self, project_id: str) -> Project | None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE projects
                SET updated_at = ?, last_mentioned_at = ?
                WHERE id = ?
                """,
                (datetime_to_iso(now), datetime_to_iso(now), project_id),
            )
        return self.get_project(project_id)

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
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    aliases TEXT NOT NULL DEFAULT '[]',
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_mentioned_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_projects_user_status
                ON projects (user_id, status)
                """
            )

    @staticmethod
    def _project_to_row(project: Project) -> tuple[object, ...]:
        return (
            project.id,
            project.user_id,
            project.name,
            json.dumps(project.aliases, ensure_ascii=False),
            project.description,
            project.status,
            datetime_to_iso(project.created_at),
            datetime_to_iso(project.updated_at),
            datetime_to_iso(project.last_mentioned_at) if project.last_mentioned_at else None,
        )

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> Project:
        return Project(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            aliases=json.loads(row["aliases"]),
            description=row["description"],
            status=row["status"],
            created_at=datetime_from_iso(row["created_at"]),
            updated_at=datetime_from_iso(row["updated_at"]),
            last_mentioned_at=(
                datetime_from_iso(row["last_mentioned_at"]) if row["last_mentioned_at"] else None
            ),
        )
