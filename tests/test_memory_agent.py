import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryAgent, MemoryLayer


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        context = "\n".join(message["content"] for message in kwargs["messages"])
        if "XGBoost" in context:
            return FakeResponse("The churn project uses XGBoost.")
        return FakeResponse("No memory found.")


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class MemoryAgentTest(unittest.TestCase):
    def test_agent_records_assistant_reply_and_uses_context(self):
        with TemporaryDirectory() as tmpdir:
            client = FakeClient()
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            agent = MemoryAgent(memory=memory, client=client)

            agent.chat(
                user_id="user_1",
                session_id="session_1",
                content="I am working on a churn project using XGBoost.",
            )
            reply = agent.chat(
                user_id="user_1",
                session_id="session_1",
                content="Which model does the churn project use?",
            )

            self.assertEqual(reply, "The churn project uses XGBoost.")
            messages = memory.recall_store.list_messages("user_1", "session_1")
            self.assertEqual(messages[-1].role, "assistant")
            self.assertIn("Retrieved Memory Context", client.chat.completions.calls[-1]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
