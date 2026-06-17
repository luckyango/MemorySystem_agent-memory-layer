# Agent Memory Layer

A hybrid memory layer for AI agents.

This project is moving from a small MemGPT-inspired prototype into a production-style
memory system that can be reused by different agents.

## Goal

The system separates memory into practical layers:

- **Recall memory**: complete raw messages and tool events for audit and replay.
- **Session memory**: short-term state for the current conversation or task.
- **Structured long-term memory**: extracted facts, preferences, goals, constraints, and project context.
- **Project-scoped memory**: memories grouped by project so multiple user projects do not get mixed together.
- **Retrieval context**: relevant memories selected for the current query and injected into the agent prompt.

Vector databases or Mem0 can be added later as retrieval backends. The core project keeps its own
schemas, source-message links, and update policy so memories remain inspectable and portable.

## Current Status

The repository currently contains:

- `mini_memGPT.py`: the legacy single-file prototype.
- `agent_memory/`: the new package skeleton and shared data schemas.
- `agent_memory/stores/recall_store.py`: SQLite storage for raw recall messages.

## Planned Milestones

1. Define schemas and package structure.
2. Implement SQLite recall storage for raw messages.
3. Implement structured memory and project stores.
4. Add memory extraction and conflict-aware update decisions.
5. Add retrieval and context building.
6. Add optional vector/Mem0 backends and evaluation demos.

## Recall Store Example

```python
from agent_memory.schemas import Message
from agent_memory.stores import SQLiteRecallStore

store = SQLiteRecallStore("memory.sqlite3")
store.add_message(
    Message(
        user_id="user_1",
        session_id="session_1",
        role="user",
        content="I am working on a customer churn project with XGBoost.",
    )
)

matches = store.search_messages("user_1", "XGBoost")
```

Run the demo from the repository root:

```bash
python -m examples.recall_store_demo
```
