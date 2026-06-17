import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory.schemas import SessionState
from agent_memory.stores import SQLiteSessionStore


class SQLiteSessionStoreTest(unittest.TestCase):
    def test_upsert_and_get_session(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteSessionStore(Path(tmpdir) / "memory.sqlite3")
            state = store.upsert_session(
                SessionState(
                    user_id="user_1",
                    session_id="session_1",
                    active_project_id="proj_1",
                    current_task="Design memory system",
                    temporary_constraints=["be concise"],
                )
            )

            loaded = store.get_session("user_1", "session_1")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.active_project_id, state.active_project_id)
            self.assertEqual(loaded.temporary_constraints, ["be concise"])

    def test_update_session_creates_missing_state(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteSessionStore(Path(tmpdir) / "memory.sqlite3")

            state = store.update_session(
                user_id="user_1",
                session_id="session_1",
                current_task="Build retriever",
            )

            self.assertEqual(state.current_task, "Build retriever")
            self.assertEqual(
                store.get_session("user_1", "session_1").current_task,
                "Build retriever",
            )


if __name__ == "__main__":
    unittest.main()
