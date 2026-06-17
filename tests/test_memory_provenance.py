import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


class MemoryProvenanceTest(unittest.TestCase):
    def test_get_memory_with_sources_returns_raw_messages_and_evidence(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            message = memory.record_message(
                user_id="user_1",
                session_id="session_1",
                role="user",
                content="I prefer concise answers.",
            )
            saved = memory.remember(
                user_id="user_1",
                category="preference",
                content="The user prefers concise answers.",
                source_message_ids=[message.id],
                metadata={
                    "evidence_quote": "I prefer concise answers",
                    "write_action": "insert",
                    "write_reason": "New durable user preference.",
                },
            )

            provenance = memory.get_memory_with_sources(saved.id)

            self.assertIsNotNone(provenance)
            self.assertEqual(provenance.memory.id, saved.id)
            self.assertEqual([item.id for item in provenance.source_messages], [message.id])
            self.assertEqual(provenance.evidence_quote, "I prefer concise answers")
            self.assertEqual(provenance.write_action, "insert")

    def test_get_memory_with_sources_returns_none_for_missing_memory(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            self.assertIsNone(memory.get_memory_with_sources("missing"))


if __name__ == "__main__":
    unittest.main()
