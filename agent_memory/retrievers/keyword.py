from __future__ import annotations

import re

from agent_memory.schemas import MemoryItem, Message, RecallRetrievalResult, RetrievalResult
from agent_memory.stores import SQLiteMemoryStore, SQLiteRecallStore


class KeywordRetriever:
    """Simple keyword retriever with scope, importance, and confidence-aware scoring."""

    def __init__(self, memory_store: SQLiteMemoryStore, recall_store: SQLiteRecallStore):
        self.memory_store = memory_store
        self.recall_store = recall_store

    def retrieve_memories(
        self,
        *,
        user_id: str,
        query: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        query_terms = self._terms(query)
        candidates = self.memory_store.list_memories(
            user_id=user_id,
            scope_type=scope_type,
            scope_id=scope_id,
            limit=200,
        )
        results = [
            result
            for result in (
                self._score_memory(memory=memory, query_terms=query_terms)
                for memory in candidates
            )
            if result.score > 0
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:limit]

    def retrieve_recall(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 3,
    ) -> list[RecallRetrievalResult]:
        query_terms = self._terms(query)
        if not query_terms:
            return []
        messages = self.recall_store.list_messages(user_id=user_id, limit=200)
        results = [
            result
            for result in (
                self._score_message(message=message, query_terms=query_terms)
                for message in messages
            )
            if result.score > 0
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:limit]

    def _score_memory(self, *, memory: MemoryItem, query_terms: set[str]) -> RetrievalResult:
        content_terms = self._terms(" ".join([memory.content, *memory.entities]))
        overlap = query_terms.intersection(content_terms)
        if not overlap:
            return RetrievalResult(memory=memory, score=0.0, reason="No keyword overlap.")
        overlap_score = len(overlap) / max(len(query_terms), 1)
        score = overlap_score + (memory.importance * 0.25) + (memory.confidence * 0.15)
        reason = f"Matched terms: {', '.join(sorted(overlap))}."
        return RetrievalResult(memory=memory, score=round(score, 4), reason=reason)

    def _score_message(
        self,
        *,
        message: Message,
        query_terms: set[str],
    ) -> RecallRetrievalResult:
        content_terms = self._terms(message.content)
        overlap = query_terms.intersection(content_terms)
        if not overlap:
            return RecallRetrievalResult(message=message, score=0.0, reason="No keyword overlap.")
        score = len(overlap) / max(len(query_terms), 1)
        reason = f"Matched terms: {', '.join(sorted(overlap))}."
        return RecallRetrievalResult(message=message, score=round(score, 4), reason=reason)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            term.casefold()
            for term in re.findall(r"[A-Za-z0-9_.+-]+", text)
            if len(term) > 1
        }
