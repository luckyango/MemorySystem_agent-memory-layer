import unittest

from agent_memory.conflicts import RuleBasedConflictResolver
from agent_memory.schemas import MemoryCandidate, MemoryItem


class RuleBasedConflictResolverTest(unittest.TestCase):
    def test_ignores_exact_duplicate(self):
        resolver = RuleBasedConflictResolver()
        candidate = MemoryCandidate(
            category="preference",
            content="The user prefers concise answers.",
        )
        existing = MemoryItem(
            user_id="user_1",
            scope_type="user",
            scope_id="global",
            category="preference",
            content="The user prefers concise answers.",
        )

        decision = resolver.decide(candidate=candidate, related_memories=[existing])

        self.assertEqual(decision.action, "ignore")
        self.assertEqual(decision.existing_memory, existing)

    def test_updates_when_entities_overlap(self):
        resolver = RuleBasedConflictResolver()
        candidate = MemoryCandidate(
            category="project",
            content="The project now uses LightGBM alongside XGBoost.",
            entities=["XGBoost", "LightGBM"],
        )
        existing = MemoryItem(
            user_id="user_1",
            scope_type="project",
            scope_id="proj_1",
            category="project",
            content="The project uses XGBoost.",
            entities=["XGBoost"],
        )

        decision = resolver.decide(candidate=candidate, related_memories=[existing])

        self.assertEqual(decision.action, "update")
        self.assertIn("LightGBM", decision.merged_content)

    def test_inserts_new_candidate(self):
        resolver = RuleBasedConflictResolver()
        candidate = MemoryCandidate(
            category="goal",
            content="The user wants to build a memory evaluation suite.",
        )

        decision = resolver.decide(candidate=candidate, related_memories=[])

        self.assertEqual(decision.action, "insert")


if __name__ == "__main__":
    unittest.main()
