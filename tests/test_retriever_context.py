import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


class RetrieverContextTest(unittest.TestCase):
    def test_retrieve_context_returns_structured_and_recall_matches(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            project = memory.create_project(
                user_id="user_1",
                name="Customer Churn",
                aliases=["churn project"],
            )
            message = memory.record_message(
                user_id="user_1",
                session_id="session_1",
                role="user",
                content="The churn project uses XGBoost.",
            )
            saved = memory.remember_for_project(
                user_id="user_1",
                project_id=project.id,
                content="The customer churn project uses XGBoost.",
                source_message_ids=[message.id],
                entities=["XGBoost"],
                importance=0.8,
            )

            context = memory.retrieve_context(
                user_id="user_1",
                query="Which project uses XGBoost?",
                scope_type="project",
                scope_id=project.id,
            )

            self.assertEqual(context.memories[0].memory.id, saved.id)
            self.assertEqual(context.recall_messages[0].message.id, message.id)

    def test_build_context_formats_prompt_block(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            memory.remember(
                user_id="user_1",
                category="preference",
                content="The user prefers concise answers.",
                importance=0.9,
            )

            block = memory.build_context(
                user_id="user_1",
                query="What answer style does the user prefer?",
            )

            self.assertIn("## Retrieved Memory Context", block)
            self.assertIn("The user prefers concise answers.", block)
            self.assertIn("Structured Memories", block)


if __name__ == "__main__":
    unittest.main()
