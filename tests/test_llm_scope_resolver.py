import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory.schemas import Project
from agent_memory.stores import SQLiteProjectStore

pydantic_available = find_spec("pydantic") is not None
if pydantic_available:
    from agent_memory.resolvers.llm import LLMScopeResolver, ScopeResolutionModel


class FakeMessage:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeChoice:
    def __init__(self, parsed):
        self.message = FakeMessage(parsed)


class FakeResponse:
    def __init__(self, parsed):
        self.choices = [FakeChoice(parsed)]


class FakeCompletions:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.parsed)


class FakeChat:
    def __init__(self, parsed):
        self.completions = FakeCompletions(parsed)


class FakeBeta:
    def __init__(self, parsed):
        self.chat = FakeChat(parsed)


class FakeClient:
    def __init__(self, parsed):
        self.beta = FakeBeta(parsed)


class LLMScopeResolverTest(unittest.TestCase):
    @unittest.skipUnless(pydantic_available, "pydantic is not installed")
    def test_resolves_existing_project_from_llm_output(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteProjectStore(Path(tmpdir) / "memory.sqlite3")
            project = store.add_project(
                Project(user_id="user_1", name="Customer Churn", aliases=["churn"])
            )
            parsed = ScopeResolutionModel.model_validate(
                {
                    "kind": "existing_project",
                    "confidence": 0.92,
                    "reason": "The message refers to the churn project.",
                    "project_id": project.id,
                    "matched_text": "churn project",
                }
            )
            resolver = LLMScopeResolver(project_store=store, client=FakeClient(parsed))

            resolution = resolver.resolve(
                user_id="user_1",
                text="The churn project now uses LightGBM.",
            )

            self.assertEqual(resolution.kind, "existing_project")
            self.assertEqual(resolution.scope_id, project.id)
            call = resolver.client.beta.chat.completions.calls[0]
            self.assertIs(call["response_format"], ScopeResolutionModel)

    @unittest.skipUnless(pydantic_available, "pydantic is not installed")
    def test_resolve_or_create_project_creates_new_project_from_llm_output(self):
        with TemporaryDirectory() as tmpdir:
            store = SQLiteProjectStore(Path(tmpdir) / "memory.sqlite3")
            parsed = ScopeResolutionModel.model_validate(
                {
                    "kind": "new_project",
                    "confidence": 0.86,
                    "reason": "The user introduced a new project.",
                    "suggested_project_name": "Knowledge Base QA",
                    "matched_text": "knowledge base QA project",
                }
            )
            resolver = LLMScopeResolver(project_store=store, client=FakeClient(parsed))

            resolution = resolver.resolve_or_create_project(
                user_id="user_1",
                text="I am starting a knowledge base QA project.",
            )

            self.assertEqual(resolution.kind, "existing_project")
            self.assertEqual(resolution.project.name, "Knowledge Base QA")
            self.assertEqual(len(store.list_projects("user_1")), 1)


if __name__ == "__main__":
    unittest.main()
