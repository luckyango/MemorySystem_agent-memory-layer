import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from agent_memory.schemas import Message
from agent_memory.stores import SQLiteRecallStore


class SQLiteRecallStoreTest(unittest.TestCase):
    def test_add_and_list_messages(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteRecallStore(Path(tmpdir) / "memory.sqlite3")
            first = store.add_message(
                Message(user_id="user_1", session_id="session_1", role="user", content="hello")
            )
            second = store.add_message(
                Message(user_id="user_1", session_id="session_1", role="assistant", content="hi")
            )

            messages = store.list_messages("user_1", "session_1")

            self.assertEqual([message.id for message in messages], [first.id, second.id])
            self.assertEqual([message.content for message in messages], ["hello", "hi"])

    def test_search_messages_by_content(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteRecallStore(Path(tmpdir) / "memory.sqlite3")
            store.add_messages(
                [
                    Message(
                        user_id="user_1",
                        session_id="session_1",
                        role="user",
                        content="The churn project uses XGBoost.",
                    ),
                    Message(
                        user_id="user_1",
                        session_id="session_1",
                        role="user",
                        content="The support bot uses LangGraph.",
                    ),
                ]
            )

            results = store.search_messages("user_1", "XGBoost")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].content, "The churn project uses XGBoost.")

    def test_get_message(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteRecallStore(Path(tmpdir) / "memory.sqlite3")
            saved = store.add_message(
                Message(
                    user_id="user_1",
                    session_id="session_1",
                    role="tool",
                    content='{"ok": true}',
                    metadata={"tool_name": "demo_tool"},
                )
            )

            loaded = store.get_message(saved.id)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.id, saved.id)
            self.assertEqual(loaded.metadata, {"tool_name": "demo_tool"})


if __name__ == "__main__":
    unittest.main()
