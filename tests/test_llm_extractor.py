import unittest
from importlib.util import find_spec

from agent_memory.schemas import ScopeResolution

pydantic_available = find_spec("pydantic") is not None
if pydantic_available:
    from agent_memory.extractors import LLMMemoryExtractor, RuleBasedMemoryExtractor
    from agent_memory.extractors.llm import MemoryExtractionResult


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
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)


class FakeChat:
    def __init__(self, results):
        self.completions = FakeCompletions(results)


class FakeBeta:
    def __init__(self, results):
        self.chat = FakeChat(results)


class FakeClient:
    def __init__(self, results):
        self.beta = FakeBeta(results)


class LLMMemoryExtractorTest(unittest.TestCase):
    @unittest.skipUnless(pydantic_available, "pydantic is not installed")
    def test_extracts_valid_candidates_from_pydantic_parse(self):
        parsed = MemoryExtractionResult.model_validate(
            {
                "should_save": True,
                "reason": "The user stated a durable preference.",
                "candidates": [
                    {
                        "category": "preference",
                        "content": "The user prefers concise answers.",
                        "confidence": 0.9,
                        "importance": 0.8,
                        "evidence_quote": "I prefer concise answers",
                        "entities": ["concise answers"],
                    }
                ],
            }
        )
        client = FakeClient([parsed])
        extractor = LLMMemoryExtractor(client=client)
        scope = self._user_scope()

        candidates = extractor.extract(text="I prefer concise answers.", scope=scope)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].category, "preference")
        self.assertEqual(candidates[0].metadata["extractor"], "llm")
        self.assertEqual(candidates[0].metadata["evidence_quote"], "I prefer concise answers")
        call = client.beta.chat.completions.calls[0]
        self.assertIs(call["response_format"], MemoryExtractionResult)

    @unittest.skipUnless(pydantic_available, "pydantic is not installed")
    def test_retries_when_evidence_quote_is_not_supported(self):
        invalid = MemoryExtractionResult.model_validate(
            {
                "should_save": True,
                "reason": "Unsupported quote.",
                "candidates": [
                    {
                        "category": "preference",
                        "content": "The user prefers concise answers.",
                        "confidence": 0.9,
                        "importance": 0.8,
                        "evidence_quote": "likes long answers",
                    }
                ],
            }
        )
        valid = MemoryExtractionResult.model_validate(
            {
                "should_save": True,
                "reason": "Supported quote.",
                "candidates": [
                    {
                        "category": "preference",
                        "content": "The user prefers concise answers.",
                        "confidence": 0.9,
                        "importance": 0.8,
                        "evidence_quote": "I prefer concise answers",
                    }
                ],
            }
        )
        client = FakeClient([invalid, valid])
        extractor = LLMMemoryExtractor(client=client, max_retries=1)

        candidates = extractor.extract(text="I prefer concise answers.", scope=self._user_scope())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(client.beta.chat.completions.calls), 2)
        retry_prompt = client.beta.chat.completions.calls[1]["messages"][1]["content"]
        self.assertIn("previous_validation_error", retry_prompt)

    @unittest.skipUnless(pydantic_available, "pydantic is not installed")
    def test_rejects_project_memory_without_entities_and_uses_fallback(self):
        invalid = MemoryExtractionResult.model_validate(
            {
                "should_save": True,
                "reason": "Project stack.",
                "candidates": [
                    {
                        "category": "project",
                        "content": "The project uses XGBoost.",
                        "confidence": 0.8,
                        "importance": 0.7,
                        "evidence_quote": "project using XGBoost",
                        "entities": [],
                    }
                ],
            }
        )
        extractor = LLMMemoryExtractor(
            client=FakeClient([invalid]),
            fallback=RuleBasedMemoryExtractor(),
            max_retries=0,
        )
        scope = ScopeResolution(
            kind="existing_project",
            scope_type="project",
            scope_id="proj_1",
            confidence=0.8,
            reason="test",
        )

        candidates = extractor.extract(text="project using XGBoost", scope=scope)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].metadata["extractor"], "rule_based")

    @unittest.skipUnless(pydantic_available, "pydantic is not installed")
    def test_falls_back_when_client_raises(self):
        extractor = LLMMemoryExtractor(
            client=FakeClient([RuntimeError("boom")]),
            fallback=RuleBasedMemoryExtractor(),
        )

        candidates = extractor.extract(text="I prefer concise answers.", scope=self._user_scope())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].metadata["extractor"], "rule_based")

    @staticmethod
    def _user_scope():
        return ScopeResolution(
            kind="user",
            scope_type="user",
            scope_id="global",
            confidence=0.8,
            reason="test",
        )


if __name__ == "__main__":
    unittest.main()
