"""Storage adapters for recall, session, project, and long-term memories."""

from agent_memory.stores.memory_store import SQLiteMemoryStore
from agent_memory.stores.project_store import SQLiteProjectStore
from agent_memory.stores.recall_store import SQLiteRecallStore
from agent_memory.stores.session_store import SQLiteSessionStore

__all__ = [
    "ChromaMemoryStore",
    "SQLiteMemoryStore",
    "SQLiteProjectStore",
    "SQLiteRecallStore",
    "SQLiteSessionStore",
]


def __getattr__(name: str):
    if name == "ChromaMemoryStore":
        from agent_memory.stores.chroma_store import ChromaMemoryStore

        return ChromaMemoryStore
    raise AttributeError(name)
