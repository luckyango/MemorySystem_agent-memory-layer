import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


class MemoryLayerConflictTest(unittest.TestCase):
    def test_process_user_message_ignores_duplicate_memory(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            memory.process_user_message(
                user_id="user_1",
                session_id="session_1",
                content="I prefer concise answers.",
            )
            _, saved = memory.process_user_message(
                user_id="user_1",
                session_id="session_1",
                content="I prefer concise answers.",
            )

            self.assertEqual(saved, [])
            self.assertEqual(len(memory.list_user_memories(user_id="user_1")), 1)


if __name__ == "__main__":
    unittest.main()
