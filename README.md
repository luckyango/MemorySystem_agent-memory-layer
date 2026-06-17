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
- `agent_memory/memory_layer.py`: high-level API for recall, projects, and structured memories.
- `agent_memory/resolvers/scope_resolver.py`: rule-based scope resolver for project/user/session routing.
- `agent_memory/stores/recall_store.py`: SQLite storage for raw recall messages.
- `agent_memory/stores/project_store.py`: SQLite project registry for multi-project users.
- `agent_memory/stores/memory_store.py`: SQLite storage for structured long-term memories.
- `agent_memory/extractors/rule_based.py`: deterministic baseline extractor for obvious memories.
- `agent_memory/extractors/llm.py`: LLM-backed extractor with JSON validation and fallback support.
- `agent_memory/conflicts/rule_based.py`: deterministic baseline for duplicate and simple update decisions.

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

## Structured Memory Example

```python
from agent_memory.schemas import MemoryItem, Project
from agent_memory.stores import SQLiteMemoryStore, SQLiteProjectStore

project_store = SQLiteProjectStore("memory.sqlite3")
memory_store = SQLiteMemoryStore("memory.sqlite3")

project = project_store.add_project(
    Project(user_id="user_1", name="Customer Churn", aliases=["churn project"])
)

memory_store.add_memory(
    MemoryItem(
        user_id="user_1",
        scope_type="project",
        scope_id=project.id,
        category="project",
        content="The customer churn project uses XGBoost.",
    )
)
```

## Memory Layer Example

```python
from agent_memory import MemoryLayer

memory = MemoryLayer("memory.sqlite3")

message = memory.record_message(
    user_id="user_1",
    session_id="session_1",
    role="user",
    content="I am working on a customer churn project with XGBoost.",
)

project = memory.create_project(
    user_id="user_1",
    name="Customer Churn",
    aliases=["churn project"],
)

memory.remember_for_project(
    user_id="user_1",
    project_id=project.id,
    content="The customer churn project uses XGBoost.",
    source_message_ids=[message.id],
)
```

## Scope Resolution Example

```python
resolution = memory.resolve_scope(
    user_id="user_1",
    text="The churn project now uses LightGBM.",
)

if resolution.kind == "existing_project":
    print(resolution.project.name)
```

## Processing A User Message

```python
message, saved_memories = memory.process_user_message(
    user_id="user_1",
    session_id="session_1",
    content="I am working on a customer churn project using XGBoost.",
)
```

## LLM Memory Extraction

```python
from agent_memory import MemoryLayer
from agent_memory.extractors import LLMMemoryExtractor, RuleBasedMemoryExtractor

extractor = LLMMemoryExtractor(fallback=RuleBasedMemoryExtractor())
memory = MemoryLayer("memory.sqlite3", extractor=extractor)
```

The LLM extractor uses API-native structured outputs through Pydantic parsing and then
applies business validation. Every extracted memory candidate must include an
`evidence_quote` copied from the source user message, and project memories must include
at least one entity.

## Conflict-Aware Writes

`process_user_message()` now checks related memories before writing:

- exact duplicate candidates are ignored
- candidates with overlapping entities in the same scope/category update the existing memory
- otherwise the candidate is inserted as a new memory
