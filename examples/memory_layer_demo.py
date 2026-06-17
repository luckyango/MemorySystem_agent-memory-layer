from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
        user_id = "user_1"
        session_id = "session_1"

        user_message = memory.record_message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content="I am working on a customer churn project with XGBoost.",
        )
        project = memory.create_project(
            user_id=user_id,
            name="Customer Churn",
            aliases=["churn project"],
        )
        memory.remember_for_project(
            user_id=user_id,
            project_id=project.id,
            content="The customer churn project uses XGBoost.",
            source_message_ids=[user_message.id],
            entities=["XGBoost"],
            importance=0.8,
        )

        for item in memory.search_memories(user_id=user_id, query="XGBoost"):
            print(f"{item.scope_type}:{item.scope_id} -> {item.content}")


if __name__ == "__main__":
    main()
