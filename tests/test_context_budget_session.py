import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import ContextBudget, MemoryLayer, RetrievalConfig


class ContextBudgetSessionTest(unittest.TestCase):
    def test_build_context_includes_session_state_and_active_project(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            project = memory.create_project(
                user_id="user_1",
                name="Customer Churn",
                aliases=["churn"],
            )
            memory.update_session_state(
                user_id="user_1",
                session_id="session_1",
                active_project_id=project.id,
                current_task="Compare gradient boosting models",
                temporary_constraints=["keep answers concise"],
            )
            memory.remember_for_project(
                user_id="user_1",
                project_id=project.id,
                content="The customer churn project uses XGBoost.",
                entities=["XGBoost"],
            )

            block = memory.build_context(
                user_id="user_1",
                session_id="session_1",
                query="Which model does the churn project use?",
            )

            self.assertIn("### Session State", block)
            self.assertIn("Active Project: Customer Churn", block)
            self.assertIn("Current Task: Compare gradient boosting models", block)
            self.assertIn("keep answers concise", block)
            self.assertIn("XGBoost", block)

    def test_context_budget_truncates_prompt_block(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            memory.remember(
                user_id="user_1",
                category="preference",
                content="The user prefers " + "very " * 100 + "concise answers.",
            )

            block = memory.build_context(
                user_id="user_1",
                query="What does the user prefer?",
                budget=ContextBudget(max_context_chars=180, memory_chars=120),
            )

            self.assertLessEqual(len(block), 180 + len("\n[context truncated]"))
            self.assertIn("truncated", block)

    def test_category_filter_limits_retrieved_memories(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(
                Path(tmpdir) / "memory.sqlite3",
                retrieval_config=RetrievalConfig(categories=["preference"]),
            )
            memory.remember(
                user_id="user_1",
                category="preference",
                content="The user prefers concise answers.",
            )
            memory.remember(
                user_id="user_1",
                category="project",
                scope_type="project",
                scope_id="proj_1",
                content="The project uses concise summaries.",
            )

            context = memory.retrieve_context(
                user_id="user_1",
                query="concise",
            )

            self.assertEqual(len(context.memories), 1)
            self.assertEqual(context.memories[0].memory.category, "preference")


if __name__ == "__main__":
    unittest.main()
