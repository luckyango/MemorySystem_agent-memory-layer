from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_memory.schemas import MemoryItem


class ChromaMemoryStore:
    """ChromaDB-backed vector index for structured memories."""

    def __init__(
        self,
        *,
        path: str | Path,
        collection_name: str = "agent_memories",
        embedding_function: Any | None = None,
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Install chromadb to use ChromaMemoryStore: python -m pip install chromadb"
            ) from exc

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    def upsert_memory(self, memory: MemoryItem) -> None:
        self.collection.upsert(
            ids=[memory.id],
            documents=[memory.content],
            metadatas=[
                {
                    "user_id": memory.user_id,
                    "scope_type": memory.scope_type,
                    "scope_id": memory.scope_id,
                    "category": memory.category,
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                }
            ],
        )

    def delete_memory(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def query(
        self,
        *,
        user_id: str,
        query: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        where: dict[str, Any] = {"user_id": user_id}
        if scope_type is not None:
            where["scope_type"] = scope_type
        if scope_id is not None:
            where["scope_id"] = scope_id

        result = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where,
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits = []
        for memory_id, distance in zip(ids, distances):
            # Chroma returns lower distance for better matches. Convert to a bounded similarity score.
            score = 1.0 / (1.0 + float(distance))
            hits.append((memory_id, score))
        return hits
