from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory.schemas import Message
from agent_memory.stores import SQLiteRecallStore


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        store = SQLiteRecallStore(Path(tmpdir) / "memory.sqlite3")
        store.add_message(
            Message(
                user_id="user_1",
                session_id="session_1",
                role="user",
                content="I am working on a customer churn project with XGBoost.",
            )
        )
        store.add_message(
            Message(
                user_id="user_1",
                session_id="session_1",
                role="assistant",
                content="Got it. I will remember the project context.",
            )
        )

        for message in store.search_messages("user_1", "XGBoost"):
            print(f"{message.role}: {message.content}")


if __name__ == "__main__":
    main()
