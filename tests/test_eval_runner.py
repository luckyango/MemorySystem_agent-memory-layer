import unittest

from agent_memory.evaluation import EvalCase, MemoryEvalRunner


class MemoryEvalRunnerTest(unittest.TestCase):
    def test_eval_case_passes_when_expected_memory_and_context_are_present(self):
        result = MemoryEvalRunner().run_case(
            EvalCase(
                name="preference",
                inputs=["I prefer concise answers."],
                query="What answer style does the user prefer?",
                expected_memory_substrings=["The user prefers concise answers."],
                expected_context_substrings=["The user prefers concise answers."],
                expected_memory_count=1,
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.failures, [])

    def test_eval_case_reports_failures(self):
        result = MemoryEvalRunner().run_case(
            EvalCase(
                name="missing",
                inputs=["Sounds good."],
                query="What should be remembered?",
                expected_memory_substrings=["nonexistent"],
            )
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 1)


if __name__ == "__main__":
    unittest.main()
