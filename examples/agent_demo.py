from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryAgent, MemoryLayer


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def create(self, *, model: str, messages: list[dict]):
        context = "\n".join(message["content"] for message in messages if "XGBoost" in message["content"])
        if "XGBoost" in context:
            return FakeResponse("You said the customer churn project uses XGBoost.")
        return FakeResponse("I do not have enough memory context for that yet.")


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
        agent = MemoryAgent(memory=memory, client=FakeClient())

        agent.chat(
            user_id="user_1",
            session_id="session_1",
            content="I am working on a customer churn project using XGBoost.",
        )
        reply = agent.chat(
            user_id="user_1",
            session_id="session_1",
            content="Which model did I say the churn project uses?",
        )
        print(reply)


if __name__ == "__main__":
    main()
