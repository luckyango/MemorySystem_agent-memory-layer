import unittest
from importlib.util import find_spec

from agent_memory.schemas import MemoryCandidate, MemoryItem

pydantic_available = find_spec("pydantic") is not None
if pydantic_available:
    from agent_memory.conflicts.llm import LLMConflictResolver, MemoryWriteDecisionModel


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


class LLMConflictResolverTest(unittest.TestCase):
    @unittest.skipUnless(pydantic_available, "pydantic is not installed")
    def test_uses_llm_update_decision(self):
        existing = MemoryItem(
            user_id="user_1",
            scope_type="user",
            scope_id="global",
            category="preference",
            content="The user prefers concise answers.",
        )
        candidate = MemoryCandidate(
            category="preference",
            content="The user prefers concise but detailed answers.",
        )
        parsed = MemoryWriteDecisionModel.model_validate(
            {
                "action": "update",
                "reason": "The candidate refines the existing preference.",
                "existing_memory_id": existing.id,
                "merged_content": "The user prefers concise but sufficiently detailed answers.",
            }
        )
        resolver = LLMConflictResolver(client=FakeClient(parsed))

        decision = resolver.decide(candidate=candidate, related_memories=[existing])

        self.assertEqual(decision.action, "update")
        self.assertEqual(decision.existing_memory.id, existing.id)
        self.assertIn("sufficiently detailed", decision.merged_content)
        call = resolver.client.beta.chat.completions.calls[0]
        self.assertIs(call["response_format"], MemoryWriteDecisionModel)

    @unittest.skipUnless(pydantic_available, "pydantic is not installed")
    def test_falls_back_when_llm_references_unknown_memory(self):
        existing = MemoryItem(
            user_id="user_1",
            scope_type="user",
            scope_id="global",
            category="preference",
            content="The user prefers concise answers.",
        )
        candidate = MemoryCandidate(
            category="preference",
            content="The user prefers concise answers.",
        )
        parsed = MemoryWriteDecisionModel.model_validate(
            {
                "action": "ignore",
                "reason": "Duplicate.",
                "existing_memory_id": "missing",
            }
        )
        resolver = LLMConflictResolver(client=FakeClient(parsed))

        decision = resolver.decide(candidate=candidate, related_memories=[existing])

        self.assertEqual(decision.action, "ignore")
        self.assertEqual(decision.existing_memory.id, existing.id)


if __name__ == "__main__":
    unittest.main()
