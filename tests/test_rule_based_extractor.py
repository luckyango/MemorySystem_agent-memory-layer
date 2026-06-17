import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer
from agent_memory.extractors import RuleBasedMemoryExtractor


class RuleBasedMemoryExtractorTest(unittest.TestCase):
    def test_extracts_user_preference(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            scope = memory.resolve_scope(user_id="user_1", text="I prefer concise answers.")

            candidates = RuleBasedMemoryExtractor().extract(
                text="I prefer concise answers.",
                scope=scope,
            )

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].category, "preference")
            self.assertIn("concise answers", candidates[0].content)

    def test_process_user_message_records_and_saves_memory(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            message, saved = memory.process_user_message(
                user_id="user_1",
                session_id="session_1",
                content="I am working on a customer churn project using XGBoost.",
            )

            self.assertEqual(len(saved), 2)
            self.assertEqual(message.content, "I am working on a customer churn project using XGBoost.")
            self.assertEqual(saved[0].source_message_ids, [message.id])
            self.assertEqual(saved[0].scope_type, "project")

    def test_unknown_scope_does_not_save_memory(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            _, saved = memory.process_user_message(
                user_id="user_1",
                session_id="session_1",
                content="Sounds good.",
            )

            self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
