from agent_memory.evaluation import EvalCase, MemoryEvalRunner


CASES = [
    EvalCase(
        name="preference_memory",
        inputs=["I prefer concise answers."],
        query="What answer style does the user prefer?",
        expected_memory_substrings=["The user prefers concise answers."],
        expected_context_substrings=["The user prefers concise answers."],
        expected_memory_count=1,
    ),
    EvalCase(
        name="project_stack_memory",
        inputs=["I am working on a churn project using XGBoost."],
        query="Which model does the churn project use?",
        expected_memory_substrings=["The project uses XGBoost."],
        expected_context_substrings=["XGBoost"],
    ),
    EvalCase(
        name="duplicate_preference_is_ignored",
        inputs=["I prefer concise answers.", "I prefer concise answers."],
        query="What answer style does the user prefer?",
        expected_memory_substrings=["The user prefers concise answers."],
        expected_context_substrings=["The user prefers concise answers."],
        expected_memory_count=1,
    ),
]


def main() -> None:
    runner = MemoryEvalRunner()
    results = runner.run(CASES)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        print("  inputs:")
        for item in result.inputs:
            print(f"    - {item}")
        print(f"  query: {result.query}")
        print("  saved_memories:")
        for memory in result.saved_memories:
            print(f"    - {memory}")
        if result.failures:
            print("  failures:")
            for failure in result.failures:
                print(f"    - {failure}")
        print()

    failed = [result for result in results if not result.passed]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
