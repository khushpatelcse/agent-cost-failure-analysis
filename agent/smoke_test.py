# smoke_test - first OSC job: does Llama load on a GPU and produce a plan?
# Also a zero-shot baseline before any training.

import json
import sys

from agent import F1Agent, AgentError
from llama_model import LlamaModel

adapter = sys.argv[1] if len(sys.argv) > 1 else None
model = LlamaModel(adapter_path=adapter)
checker = F1Agent(model=model, tools={})

for question in [
    "Who won the 2025 Miami Grand Prix?",
    "Did the pole sitter win the 2024 Monaco Grand Prix?",
    "Who leads the 2026 drivers' championship?",
]:
    raw = model.plan_tools(question)
    print("\nQ:", question)
    print("raw plan:", raw)
    print("usage:", model.last_usage)
    try:
        plan = checker._read_plan(raw)
        for c in plan["tool_calls"]:
            checker._check_tool_call(c["tool"], c["arguments"])
        print("agent accepts plan: yes")
    except AgentError as error:
        print("agent accepts plan: NO ->", error)
