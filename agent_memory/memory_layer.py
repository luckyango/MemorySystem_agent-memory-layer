from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_memory.conflicts import RuleBasedConflictResolver
from agent_memory.context import ContextBuilder
from agent_memory.extractors import RuleBasedMemoryExtractor
from agent_memory.resolvers import RuleBasedScopeResolver
from agent_memory.retrievers import KeywordRetriever
from agent_memory.schemas import (
    MemoryCategory,
    MemoryCandidate,
    MemoryContext,
    MemoryItem,
    Message,
    MessageRole,
    Project,
    ScopeResolution,
    ScopeType,
)
from agent_memory.stores import SQLiteMemoryStore, SQLiteProjectStore, SQLiteRecallStore


class MemoryLayer:
    """High-level API that coordinates recall, project, and structured memory stores."""

    def __init__(self, db_path: str | Path, *, extractor: Any | None = None):
        self.db_path = Path(db_path)
        self.recall_store = SQLiteRecallStore(self.db_path)
        self.project_store = SQLiteProjectStore(self.db_path)
        self.memory_store = SQLiteMemoryStore(self.db_path)
        self.scope_resolver = RuleBasedScopeResolver(self.project_store)
        self.extractor = extractor or RuleBasedMemoryExtractor()
        self.conflict_resolver = RuleBasedConflictResolver()
        self.retriever = KeywordRetriever(self.memory_store, self.recall_store)
        self.context_builder = ContextBuilder()

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

    def resolve_scope(self, *, user_id: str, text: str) -> ScopeResolution:
        return self.scope_resolver.resolve(user_id=user_id, text=text)

    def resolve_or_create_project(self, *, user_id: str, text: str) -> ScopeResolution:
        return self.scope_resolver.resolve_or_create_project(user_id=user_id, text=text)

    def extract_memory_candidates(
        self,
        *,
        user_id: str,
        text: str,
        create_projects: bool = False,
    ) -> tuple[ScopeResolution, list[MemoryCandidate]]:
        if create_projects:
            scope = self.resolve_or_create_project(user_id=user_id, text=text)
        else:
            scope = self.resolve_scope(user_id=user_id, text=text)
        related_memories = self.search_memories(user_id=user_id, query=text, limit=8)
        return scope, self.extractor.extract(
            text=text,
            scope=scope,
            related_memories=related_memories,
        )

    def process_user_message(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        create_projects: bool = True,
    ) -> tuple[Message, list[MemoryItem]]:
        message = self.record_message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=content,
        )
        scope, candidates = self.extract_memory_candidates(
            user_id=user_id,
            text=content,
            create_projects=create_projects,
        )
        saved_memories = []
        if scope.kind != "unknown":
            for candidate in candidates:
                related = self.memory_store.list_memories(
                    user_id=user_id,
                    scope_type=scope.scope_type,
                    scope_id=scope.scope_id,
                    category=candidate.category,
                    limit=20,
                )
                decision = self.conflict_resolver.decide(
                    candidate=candidate,
                    related_memories=related,
                )
                saved = self._apply_write_decision(
                    user_id=user_id,
                    scope=scope,
                    message=message,
                    decision=decision,
                )
                if saved is not None:
                    saved_memories.append(saved)
        return message, saved_memories

    def _apply_write_decision(
        self,
        *,
        user_id: str,
        scope: ScopeResolution,
        message: Message,
        decision,
    ) -> MemoryItem | None:
        candidate = decision.candidate
        metadata = {
            **candidate.metadata,
            "scope_reason": scope.reason,
            "write_action": decision.action,
            "write_reason": decision.reason,
        }
        if decision.action == "ignore":
            return None
        if decision.action == "update" and decision.existing_memory is not None:
            existing = decision.existing_memory
            source_ids = [*existing.source_message_ids]
            if message.id not in source_ids:
                source_ids.append(message.id)
            entities = sorted({*existing.entities, *candidate.entities}, key=str.casefold)
            updated = self.memory_store.update_memory(
                existing.id,
                content=decision.merged_content or candidate.content,
                confidence=max(existing.confidence, candidate.confidence),
                importance=max(existing.importance, candidate.importance),
                source_message_ids=source_ids,
                entities=entities,
                metadata={**existing.metadata, **metadata},
            )
            if updated is None:
                return None
            updated.source_message_ids = source_ids
            updated.entities = entities
            return updated
        return self.remember(
            user_id=user_id,
            content=candidate.content,
            category=candidate.category,
            scope_type=scope.scope_type,
            scope_id=scope.scope_id,
            confidence=candidate.confidence,
            importance=candidate.importance,
            source_message_ids=[message.id],
            entities=candidate.entities,
            metadata=metadata,
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

    def retrieve_context(
        self,
        *,
        user_id: str,
        query: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
        memory_limit: int = 5,
        recall_limit: int = 3,
    ) -> MemoryContext:
        memories = self.retriever.retrieve_memories(
            user_id=user_id,
            query=query,
            scope_type=scope_type,
            scope_id=scope_id,
            limit=memory_limit,
        )
        recall_messages = self.retriever.retrieve_recall(
            user_id=user_id,
            query=query,
            limit=recall_limit,
        )
        return MemoryContext(
            query=query,
            memories=memories,
            recall_messages=recall_messages,
        )

    def build_context(
        self,
        *,
        user_id: str,
        query: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
        memory_limit: int = 5,
        recall_limit: int = 3,
    ) -> str:
        context = self.retrieve_context(
            user_id=user_id,
            query=query,
            scope_type=scope_type,
            scope_id=scope_id,
            memory_limit=memory_limit,
            recall_limit=recall_limit,
        )
        return self.context_builder.build(context)
