import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


class MemoryLayerTest(unittest.TestCase):
    def test_record_project_and_memory_flow(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            message = memory.record_message(
                user_id="user_1",
                session_id="session_1",
                role="user",
                content="The churn project uses XGBoost.",
            )
            project = memory.create_project(
                user_id="user_1",
                name="Customer Churn",
                aliases=["churn project"],
            )
            saved = memory.remember_for_project(
                user_id="user_1",
                project_id=project.id,
                content="The churn project uses XGBoost.",
                source_message_ids=[message.id],
                entities=["XGBoost"],
            )

            project_memories = memory.list_project_memories(
                user_id="user_1",
                project_id=project.id,
            )
            recall_matches = memory.search_recall(user_id="user_1", query="XGBoost")
            memory_matches = memory.search_memories(user_id="user_1", query="XGBoost")

            self.assertEqual([item.id for item in project_memories], [saved.id])
            self.assertEqual([item.id for item in memory_matches], [saved.id])
            self.assertEqual([item.id for item in recall_matches], [message.id])

    def test_remember_global_user_preference(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            saved = memory.remember(
                user_id="user_1",
                category="preference",
                content="The user prefers concise answers.",
                importance=0.9,
            )

            user_memories = memory.list_user_memories(user_id="user_1", category="preference")

            self.assertEqual([item.id for item in user_memories], [saved.id])


if __name__ == "__main__":
    unittest.main()
