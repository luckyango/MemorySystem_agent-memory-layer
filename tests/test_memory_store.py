import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory.schemas import MemoryItem
from agent_memory.stores import SQLiteMemoryStore


class SQLiteMemoryStoreTest(unittest.TestCase):
    def test_add_and_list_project_memories(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteMemoryStore(Path(tmpdir) / "memory.sqlite3")
            memory = store.add_memory(
                MemoryItem(
                    user_id="user_1",
                    scope_type="project",
                    scope_id="proj_1",
                    category="project",
                    content="The churn project uses XGBoost.",
                    source_message_ids=["msg_1"],
                    entities=["XGBoost"],
                    importance=0.8,
                )
            )

            memories = store.list_memories("user_1", scope_type="project", scope_id="proj_1")

            self.assertEqual([item.id for item in memories], [memory.id])
            self.assertEqual(memories[0].source_message_ids, ["msg_1"])
            self.assertEqual(memories[0].entities, ["XGBoost"])

    def test_search_memories_by_content(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteMemoryStore(Path(tmpdir) / "memory.sqlite3")
            store.add_memory(
                MemoryItem(
                    user_id="user_1",
                    scope_type="user",
                    scope_id="global",
                    category="preference",
                    content="The user prefers concise answers.",
                )
            )

            results = store.search_memories("user_1", "concise")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].category, "preference")

    def test_update_memory(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteMemoryStore(Path(tmpdir) / "memory.sqlite3")
            memory = store.add_memory(
                MemoryItem(
                    user_id="user_1",
                    scope_type="user",
                    scope_id="global",
                    category="fact",
                    content="The user mainly uses Python.",
                )
            )

            updated = store.update_memory(
                memory.id,
                content="The user mainly uses Rust.",
                confidence=0.9,
                metadata={"reason": "user update"},
            )

            self.assertIsNotNone(updated)
            self.assertEqual(updated.content, "The user mainly uses Rust.")
            self.assertEqual(updated.metadata, {"reason": "user update"})


if __name__ == "__main__":
    unittest.main()
