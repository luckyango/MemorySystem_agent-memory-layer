from openai import OpenAI
import json
import time

client = OpenAI()


class LayeredMemoryAgent:
    """A layered-memory Agent inspired by MemGPT.

    Memory layers:
    1. Core Memory: always in context; stores the most important facts.
    2. Working Memory: short-term task context.
    3. Archive Memory: external storage retrieved on demand.
    """

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self.core_memory = {
            "user_name": "",
            "preferences": [],
            "key_facts": [],
            "active_goals": [],
        }
        self.working_memory = []
        self.max_working_items = 10
        self.archive_memory = []
        self.conversation = []
