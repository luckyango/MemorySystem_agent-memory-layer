from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


ScopeType = Literal["user", "project", "session"]
MessageRole = Literal["user", "assistant", "system", "tool"]
MemoryCategory = Literal[
    "profile",
    "preference",
    "project",
    "goal",
    "fact",
    "constraint",
    "relationship",
    "task_state",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(slots=True)
class Message:
    user_id: str
    session_id: str
    role: MessageRole
    content: str
    id: str = field(default_factory=lambda: new_id("msg"))
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Project:
    user_id: str
    name: str
    id: str = field(default_factory=lambda: new_id("proj"))
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    status: Literal["active", "archived"] = "active"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_mentioned_at: datetime | None = None


@dataclass(slots=True)
class MemoryItem:
    user_id: str
    scope_type: ScopeType
    scope_id: str
    category: MemoryCategory
    content: str
    id: str = field(default_factory=lambda: new_id("mem"))
    confidence: float = 1.0
    importance: float = 0.5
    source_message_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class SessionState:
    user_id: str
    session_id: str
    active_project_id: str | None = None
    current_task: str | None = None
    temporary_constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class RetrievalResult:
    memory: MemoryItem
    score: float
    reason: str = ""
