import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory.schemas import Project
from agent_memory.stores import SQLiteProjectStore


class SQLiteProjectStoreTest(unittest.TestCase):
    def test_add_and_list_projects(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteProjectStore(Path(tmpdir) / "memory.sqlite3")
            project = store.add_project(
                Project(
                    user_id="user_1",
                    name="Customer Churn",
                    aliases=["churn", "churn prediction"],
                )
            )

            projects = store.list_projects("user_1")

            self.assertEqual([item.id for item in projects], [project.id])
            self.assertEqual(projects[0].aliases, ["churn", "churn prediction"])

    def test_find_project_by_alias(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteProjectStore(Path(tmpdir) / "memory.sqlite3")
            project = store.add_project(
                Project(user_id="user_1", name="Customer Churn", aliases=["churn project"])
            )

            found = store.find_project("user_1", "What model does the churn project use?")

            self.assertIsNotNone(found)
            self.assertEqual(found.id, project.id)

    def test_touch_project_updates_last_mentioned_at(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteProjectStore(Path(tmpdir) / "memory.sqlite3")
            project = store.add_project(Project(user_id="user_1", name="Customer Churn"))

            touched = store.touch_project(project.id)

            self.assertIsNotNone(touched)
            self.assertIsNotNone(touched.last_mentioned_at)


if __name__ == "__main__":
    unittest.main()
