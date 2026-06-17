import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


class RuleBasedScopeResolverTest(unittest.TestCase):
    def test_resolves_existing_project_by_alias(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            project = memory.create_project(
                user_id="user_1",
                name="Customer Churn",
                aliases=["churn project"],
            )

            resolution = memory.resolve_scope(
                user_id="user_1",
                text="The churn project now uses LightGBM.",
            )

            self.assertEqual(resolution.kind, "existing_project")
            self.assertEqual(resolution.scope_id, project.id)

    def test_detects_new_project(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            resolution = memory.resolve_scope(
                user_id="user_1",
                text="I am starting a knowledge base project with LangGraph.",
            )

            self.assertEqual(resolution.kind, "new_project")
            self.assertEqual(resolution.scope_type, "project")
            self.assertIn("knowledge base", resolution.suggested_project_name)

    def test_resolve_or_create_project_creates_project(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            resolution = memory.resolve_or_create_project(
                user_id="user_1",
                text="I am working on a recommendation project.",
            )

            self.assertEqual(resolution.kind, "existing_project")
            self.assertEqual(resolution.scope_type, "project")
            self.assertIsNotNone(resolution.project)
            self.assertEqual(len(memory.project_store.list_projects("user_1")), 1)

    def test_detects_user_scope_preference(self):
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")

            resolution = memory.resolve_scope(
                user_id="user_1",
                text="I prefer concise answers.",
            )

            self.assertEqual(resolution.kind, "user")
            self.assertEqual(resolution.scope_id, "global")


if __name__ == "__main__":
    unittest.main()
