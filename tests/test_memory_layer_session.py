import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


class MemoryLayerSessionTest(unittest.TestCase):
    def test_process_user_message_sets_active_project(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            _, saved = memory.process_user_message(
                user_id="user_1",
                session_id="session_1",
                content="I am working on a churn project using XGBoost.",
            )
            state = memory.get_session_state(user_id="user_1", session_id="session_1")

            self.assertIsNotNone(state)
            self.assertIsNotNone(state.active_project_id)
            self.assertGreaterEqual(len(saved), 1)

    def test_update_session_state(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            state = memory.update_session_state(
                user_id="user_1",
                session_id="session_1",
                current_task="Write evals",
                temporary_constraints=["no network"],
            )

            self.assertEqual(state.current_task, "Write evals")
            self.assertEqual(state.temporary_constraints, ["no network"])


if __name__ == "__main__":
    unittest.main()
