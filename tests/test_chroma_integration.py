import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


class FakeVectorStore:
    def __init__(self):
        self.memories = {}

    def upsert_memory(self, memory):
        self.memories[memory.id] = memory

    def query(self, *, user_id, query, scope_type=None, scope_id=None, limit=5):
        hits = []
        for memory in self.memories.values():
            if memory.user_id != user_id:
                continue
            if scope_type is not None and memory.scope_type != scope_type:
                continue
            if scope_id is not None and memory.scope_id != scope_id:
                continue
            if "gradient boosting" in query.casefold() and "XGBoost" in memory.content:
                hits.append((memory.id, 0.91))
        return hits[:limit]


class ChromaIntegrationTest(unittest.TestCase):
    def test_memory_layer_syncs_to_vector_store_and_retrieves_vector_hits(self):
        with TemporaryDirectory() as tmpdir:
            vector_store = FakeVectorStore()
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3", vector_store=vector_store)
            saved = memory.remember(
                user_id="user_1",
                category="project",
                scope_type="project",
                scope_id="proj_1",
                content="The project uses XGBoost.",
                entities=["XGBoost"],
            )

            context = memory.retrieve_context(
                user_id="user_1",
                query="Which gradient boosting model is used?",
                scope_type="project",
                scope_id="proj_1",
            )

            self.assertIn(saved.id, vector_store.memories)
            self.assertEqual(context.memories[0].memory.id, saved.id)
            self.assertIn("Chroma vector similarity", context.memories[0].reason)


if __name__ == "__main__":
    unittest.main()
