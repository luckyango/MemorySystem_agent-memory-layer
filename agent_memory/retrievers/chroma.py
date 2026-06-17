from __future__ import annotations

from agent_memory.retrievers.keyword import KeywordRetriever
from agent_memory.schemas import MemoryCategory, RetrievalConfig, RetrievalResult, utc_now
from agent_memory.stores import ChromaMemoryStore, SQLiteMemoryStore, SQLiteRecallStore


class ChromaRetriever:
    """Chroma vector retriever with keyword fallback."""

    def __init__(
        self,
        *,
        memory_store: SQLiteMemoryStore,
        recall_store: SQLiteRecallStore,
        vector_store: ChromaMemoryStore,
        config: RetrievalConfig | None = None,
    ):
        self.memory_store = memory_store
        self.recall_store = recall_store
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        self.keyword_retriever = KeywordRetriever(memory_store, recall_store, config=self.config)

    def retrieve_memories(
        self,
        *,
        user_id: str,
        query: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
        categories: list[MemoryCategory] | None = None,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        vector_hits = self.vector_store.query(
            user_id=user_id,
            query=query,
            scope_type=scope_type,
            scope_id=scope_id,
            limit=limit,
        )
        allowed = set(categories if categories is not None else self.config.categories or [])
        results = []
        seen_ids = set()
        for memory_id, vector_score in vector_hits:
            memory = self.memory_store.get_memory(memory_id)
            if memory is None:
                continue
            if allowed and memory.category not in allowed:
                continue
            seen_ids.add(memory_id)
            recency_score = self._recency_score(memory)
            score = (
                vector_score * self.config.vector_weight
                + (memory.importance * self.config.importance_weight)
                + (memory.confidence * self.config.confidence_weight)
                + (recency_score * self.config.recency_weight)
            )
            results.append(
                RetrievalResult(
                    memory=memory,
                    score=round(score, 4),
                    reason=f"Chroma vector similarity={vector_score:.4f}; recency={recency_score:.2f}.",
                )
            )

        if len(results) < limit:
            for result in self.keyword_retriever.retrieve_memories(
                user_id=user_id,
                query=query,
                scope_type=scope_type,
                scope_id=scope_id,
                categories=categories,
                limit=limit,
            ):
                if result.memory.id not in seen_ids:
                    results.append(result)
                    seen_ids.add(result.memory.id)
                if len(results) >= limit:
                    break

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def retrieve_recall(self, *, user_id: str, query: str, limit: int = 3):
        return self.keyword_retriever.retrieve_recall(user_id=user_id, query=query, limit=limit)

    @staticmethod
    def _recency_score(memory) -> float:
        age_seconds = max(0.0, (utc_now() - memory.updated_at).total_seconds())
        age_days = age_seconds / 86400
        return 1.0 / (1.0 + age_days)
