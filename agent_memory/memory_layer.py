from __future__ import annotations

from pathlib import Path

from agent_memory.schemas import MemoryCategory, MemoryItem, Message, MessageRole, Project, ScopeType
from agent_memory.stores import SQLiteMemoryStore, SQLiteProjectStore, SQLiteRecallStore


class MemoryLayer:
    """High-level API that coordinates recall, project, and structured memory stores."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.recall_store = SQLiteRecallStore(self.db_path)
        self.project_store = SQLiteProjectStore(self.db_path)
        self.memory_store = SQLiteMemoryStore(self.db_path)

    def record_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        return self.recall_store.add_message(
            Message(
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                metadata=metadata or {},
            )
        )

    def create_project(
        self,
        *,
        user_id: str,
        name: str,
        aliases: list[str] | None = None,
        description: str = "",
    ) -> Project:
        return self.project_store.add_project(
            Project(
                user_id=user_id,
                name=name,
                aliases=aliases or [],
                description=description,
            )
        )

    def remember(
        self,
        *,
        user_id: str,
        content: str,
        category: MemoryCategory,
        scope_type: ScopeType = "user",
        scope_id: str = "global",
        confidence: float = 1.0,
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
        entities: list[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryItem:
        return self.memory_store.add_memory(
            MemoryItem(
                user_id=user_id,
                scope_type=scope_type,
                scope_id=scope_id,
                category=category,
                content=content,
                confidence=confidence,
                importance=importance,
                source_message_ids=source_message_ids or [],
                entities=entities or [],
                metadata=metadata or {},
            )
        )

    def remember_for_project(
        self,
        *,
        user_id: str,
        project_id: str,
        content: str,
        category: MemoryCategory = "project",
        confidence: float = 1.0,
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
        entities: list[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryItem:
        return self.remember(
            user_id=user_id,
            content=content,
            category=category,
            scope_type="project",
            scope_id=project_id,
            confidence=confidence,
            importance=importance,
            source_message_ids=source_message_ids,
            entities=entities,
            metadata=metadata,
        )

    def search_memories(self, *, user_id: str, query: str, limit: int = 20) -> list[MemoryItem]:
        return self.memory_store.search_memories(user_id=user_id, query=query, limit=limit)

    def search_recall(self, *, user_id: str, query: str, limit: int = 20) -> list[Message]:
        return self.recall_store.search_messages(user_id=user_id, query=query, limit=limit)

    def list_project_memories(
        self,
        *,
        user_id: str,
        project_id: str,
        category: MemoryCategory | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        return self.memory_store.list_memories(
            user_id=user_id,
            scope_type="project",
            scope_id=project_id,
            category=category,
            limit=limit,
        )

    def list_user_memories(
        self,
        *,
        user_id: str,
        category: MemoryCategory | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        return self.memory_store.list_memories(
            user_id=user_id,
            scope_type="user",
            scope_id="global",
            category=category,
            limit=limit,
        )
