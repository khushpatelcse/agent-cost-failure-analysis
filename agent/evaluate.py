# evaluate - compare models on the held-out planning test set
#
# This is the cost/failure analysis for the model swap. For each test question
# it records whether the plan was valid JSON, whether the agent accepted it,
# whether it matched the correct plan, the latency, and the token usage.
#
#   Gemini (laptop):  python evaluate.py --model gemini
#   Llama base (OSC): python evaluate.py --model llama
#   Llama + LoRA:     python evaluate.py --model llama --adapter outputs/f1-lora
#
# --full also runs the tools and the final answer (needs internet for OpenF1).

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from agent import F1Agent, AgentError
from tools import _normalize_name


def plan_key(plan):
    """Order-independent form of a plan, with race names normalized."""
    items = []
    for c in plan["tool_calls"]:
        args = dict(c["arguments"])
        if "race" in args:
            args["race"] = _normalize_name(args["race"])
        items.append((c["tool"], json.dumps(args, sort_keys=True)))
    return Counter(items)


def load_model(name, adapter, base):
    if name == "gemini":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
        from model import GeminiModel
        return GeminiModel()
    from llama_model import LlamaModel, DEFAULT_BASE_MODEL
    return LlamaModel(base_model=base or DEFAULT_BASE_MODEL, adapter_path=adapter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemini", "llama"], required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--base")
    parser.add_argument("--test", default="data/plan_test.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--tag", help="name for the output files")
    args = parser.parse_args()

    tag = args.tag or (args.model + ("-lora" if args.adapter else ""))
    records = [json.loads(line) for line in open(args.test, encoding="utf-8")]
    if args.limit:
        records = records[:args.limit]

    model = load_model(args.model, args.adapter, args.base)
    checker = F1Agent(model=model, tools={})

    tools = None
    if args.full:
        from tools import (get_race_results, get_qualifying_results,
                           get_fastest_lap, get_driver_standings)
        tools = {
            "get_race_results": get_race_results,
            "get_qualifying_results": get_qualifying_results,
            "get_fastest_lap": get_fastest_lap,
            "get_driver_standings": get_driver_standings,
        }

    rows = []
    for i, rec in enumerate(records, start=1):
        row = {
            "question": rec["question"],
            "held_out_wording": rec.get("held_out_wording", False),
            "valid_json": False, "valid_plan": False, "exact_match": False,
            "error_stage": "", "error": "",
            "plan_seconds": None, "input_tokens": None, "output_tokens": None,
            "raw_plan": "",
        }

        started = time.perf_counter()
        try:
            raw = model.plan_tools(rec["question"])
            row["plan_seconds"] = round(time.perf_counter() - started, 4)
            row["raw_plan"] = raw if isinstance(raw, str) else json.dumps(raw)
            usage = getattr(model, "last_usage", None) or {}
            row["input_tokens"] = usage.get("input_tokens")
            row["output_tokens"] = usage.get("output_tokens")

            try:
                plan = checker._read_plan(raw)
                row["valid_json"] = True
                for c in plan["tool_calls"]:
                    checker._check_tool_call(c["tool"], c["arguments"])
                row["valid_plan"] = bool(plan["tool_calls"])
                row["exact_match"] = plan_key(plan) == plan_key(rec["gold_plan"])
            except AgentError as error:
                row["error_stage"] = "plan_validation"
                row["error"] = str(error)
                row["valid_json"] = "valid JSON" not in str(error)

        except Exception as error:
            row["error_stage"] = "model_call"
            row["error"] = f"{type(error).__name__}: {error}"

        if args.full:
            agent = F1Agent(model=model, tools=tools)
            try:
                result = agent.run(rec["question"])
                row["tool_failures"] = result.tool_failures
                row["total_seconds"] = result.elapsed_seconds
                row["answer"] = result.answer
            except AgentError as error:
                row["tool_failures"] = None
                row["answer"] = f"AgentError: {error}"

        rows.append(row)
        print(f"[{i}/{len(records)}] match={row['exact_match']} {rec['question']}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def rate(key, subset):
        return round(sum(r[key] for r in subset) / len(subset), 3) if subset else None

    def avg(key):
        values = [r[key] for r in rows if r[key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    seen = [r for r in rows if not r["held_out_wording"]]
    held = [r for r in rows if r["held_out_wording"]]
    summary = {
        "model": tag, "n": len(rows),
        "valid_json_rate": rate("valid_json", rows),
        "valid_plan_rate": rate("valid_plan", rows),
        "exact_match_rate": rate("exact_match", rows),
        "exact_match_seen_wording": rate("exact_match", seen),
        "exact_match_held_out_wording": rate("exact_match", held),
        "avg_plan_seconds": avg("plan_seconds"),
        "avg_input_tokens": avg("input_tokens"),
        "avg_output_tokens": avg("output_tokens"),
    }
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
