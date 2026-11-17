# make_dataset - build fine-tuning data for the planning step (and optionally answers)
#
# Planning data:
#   Questions come from templates, and each template already knows the
#   correct tool plan, so labels are free and always right. No Gemini calls
#   needed. Every plan is checked with F1Agent's own validators.
#   Some templates are held out of training and only appear in the test set,
#   so the evaluation shows whether Llama generalizes to new phrasings.
#
# Answer data (optional, run on your laptop: needs internet + GEMINI_API_KEY):
#   --answers N runs the real OpenF1 tools for N training questions and has
#   Gemini write the answer. Llama learns to imitate those answers.
#
# Usage:
#   python make_dataset.py                       # planning data only
#   python make_dataset.py --answers 150         # + 150 answer examples

import argparse
import json
import random
from pathlib import Path

from agent import F1Agent
from prompts import build_plan_prompt, build_answer_prompt

YEARS = [2023, 2024, 2025, 2026]

# Name used in the question -> the same string becomes the "race" argument.
# Place names read naturally in every template ("pole at Monza in 2024") and
# match OpenF1's location / circuit fields, which tools._find_meeting searches.
# Not every race ran every year; the plan is still correct, the tool just
# reports no data, which is useful failure data too.
RACES = [
    "Bahrain", "Jeddah", "Melbourne", "Suzuka", "Shanghai", "Miami",
    "Imola", "Monaco", "Montreal", "Barcelona", "Spielberg", "Silverstone",
    "Budapest", "Spa", "Zandvoort", "Monza", "Baku", "Singapore", "Austin",
    "Mexico City", "Sao Paulo", "Las Vegas", "Qatar", "Abu Dhabi",
]

DRIVERS = [
    "Max Verstappen", "Lando Norris", "Charles Leclerc", "Lewis Hamilton",
    "Oscar Piastri", "George Russell", "Fernando Alonso", "Carlos Sainz",
]


def call(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


# Each template: (question text, function that builds the gold tool_calls).
# Placeholders: {year} {y1} {y2} {race} {race2} {driver}
TRAIN_TEMPLATES = [
    # race results
    ("Who won the {year} {race} Grand Prix?",
     lambda s: [call("get_race_results", year=s["year"], race=s["race"])]),
    ("What were the race results for {race} {year}?",
     lambda s: [call("get_race_results", year=s["year"], race=s["race"])]),
    ("Who was on the podium at the {year} {race} GP?",
     lambda s: [call("get_race_results", year=s["year"], race=s["race"])]),
    ("Where did {driver} finish at the {race} Grand Prix in {year}?",
     lambda s: [call("get_race_results", year=s["year"], race=s["race"])]),
    ("Did {driver} DNF at {race} {year}?",
     lambda s: [call("get_race_results", year=s["year"], race=s["race"])]),

    # qualifying
    ("Who got pole at {race} in {year}?",
     lambda s: [call("get_qualifying_results", year=s["year"], race=s["race"])]),
    ("Show me the qualifying results for the {year} {race} Grand Prix.",
     lambda s: [call("get_qualifying_results", year=s["year"], race=s["race"])]),
    ("Where did {driver} qualify for the {year} {race} GP?",
     lambda s: [call("get_qualifying_results", year=s["year"], race=s["race"])]),

    # fastest lap
    ("Who set the fastest lap at the {year} {race} Grand Prix?",
     lambda s: [call("get_fastest_lap", year=s["year"], race=s["race"])]),
    ("What was the quickest race lap at {race} {year}?",
     lambda s: [call("get_fastest_lap", year=s["year"], race=s["race"])]),

    # standings
    ("Who leads the {year} drivers' championship?",
     lambda s: [call("get_driver_standings", year=s["year"])]),
    ("Show me the {year} F1 driver standings.",
     lambda s: [call("get_driver_standings", year=s["year"])]),
    ("How many points does {driver} have in the {year} championship?",
     lambda s: [call("get_driver_standings", year=s["year"])]),
    ("What position is {driver} in the {year} standings?",
     lambda s: [call("get_driver_standings", year=s["year"])]),
    ("How far behind the championship leader is {driver} in {year}?",
     lambda s: [call("get_driver_standings", year=s["year"])]),
    ("Is {driver} in the top three of the {year} drivers' championship?",
     lambda s: [call("get_driver_standings", year=s["year"])]),
    ("Did {driver} score more points in {y1} or {y2}?",
     lambda s: [call("get_driver_standings", year=s["y1"]),
                call("get_driver_standings", year=s["y2"])]),

    # multi-tool
    ("Did the pole sitter win the {year} {race} Grand Prix?",
     lambda s: [call("get_qualifying_results", year=s["year"], race=s["race"]),
                call("get_race_results", year=s["year"], race=s["race"])]),
    ("How many places did {driver} gain from qualifying to the finish at {race} {year}?",
     lambda s: [call("get_qualifying_results", year=s["year"], race=s["race"]),
                call("get_race_results", year=s["year"], race=s["race"])]),
    ("Who won the {year} {race} GP, and who had the fastest lap?",
     lambda s: [call("get_race_results", year=s["year"], race=s["race"]),
                call("get_fastest_lap", year=s["year"], race=s["race"])]),
    ("Compare the winners of the {race} Grand Prix in {y1} and {y2}.",
     lambda s: [call("get_race_results", year=s["y1"], race=s["race"]),
                call("get_race_results", year=s["y2"], race=s["race"])]),
    ("Who won {race} and {race2} in {year}?",
     lambda s: [call("get_race_results", year=s["year"], race=s["race"]),
                call("get_race_results", year=s["year"], race=s["race2"])]),
    ("Compare the top of the driver standings in {y1} and {y2}.",
     lambda s: [call("get_driver_standings", year=s["y1"]),
                call("get_driver_standings", year=s["y2"])]),
]

# Never trained on: only used to test generalization to new wording.
HELD_OUT_TEMPLATES = [
    ("Which driver took the checkered flag first at {race} in {year}?",
     lambda s: [call("get_race_results", year=s["year"], race=s["race"])]),
    ("Who started P1 on the grid for the {race} race in {year}?",
     lambda s: [call("get_qualifying_results", year=s["year"], race=s["race"])]),
    ("In {year}, what was the best lap time anyone did during the {race} race?",
     lambda s: [call("get_fastest_lap", year=s["year"], race=s["race"])]),
    ("Where does {driver} rank in the {year} title fight?",
     lambda s: [call("get_driver_standings", year=s["year"])]),
    ("At {race} {year}, did {driver} finish higher than they qualified?",
     lambda s: [call("get_qualifying_results", year=s["year"], race=s["race"]),
                call("get_race_results", year=s["year"], race=s["race"])]),
]


def sample_slots(rng):
    y1, y2 = rng.sample(YEARS, 2)
    race, race2 = rng.sample(RACES, 2)
    return {
        "year": rng.choice(YEARS), "y1": y1, "y2": y2,
        "race": race, "race2": race2, "driver": rng.choice(DRIVERS),
    }


def generate(templates, n, rng, checker, seen):
    records = []
    attempts = 0
    while len(records) < n and attempts < n * 20:
        attempts += 1
        text, build_plan = rng.choice(templates)
        slots = sample_slots(rng)
        question = text.format(**slots)
        if question in seen:
            continue
        seen.add(question)

        plan = {"tool_calls": build_plan(slots)}

        # Use the agent's own validators so every label is a plan it accepts.
        checker._read_plan(plan)
        for c in plan["tool_calls"]:
            checker._check_tool_call(c["tool"], c["arguments"])

        records.append({"question": question, "gold_plan": plan})
    return records


def to_sft(prompt, completion):
    # TRL "prompt-completion" format: loss is computed on the completion only.
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": completion}],
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):>5} rows -> {path}")


def build_answer_examples(records, n):
    """Run the real tools, then have Gemini write the reference answer."""
    import time
    from pathlib import Path as _P
    from dotenv import load_dotenv
    load_dotenv(_P(__file__).resolve().parent.parent / ".env", override=True)

    from model import GeminiModel
    from tools import (get_race_results, get_qualifying_results,
                       get_fastest_lap, get_driver_standings)

    tools = {
        "get_race_results": get_race_results,
        "get_qualifying_results": get_qualifying_results,
        "get_fastest_lap": get_fastest_lap,
        "get_driver_standings": get_driver_standings,
    }
    gemini = GeminiModel()
    rows = []

    for i, rec in enumerate(records[:n], start=1):
        results = []
        for step, c in enumerate(rec["gold_plan"]["tool_calls"], start=1):
            output = tools[c["tool"]](**c["arguments"])
            results.append(F1Agent._format_tool_result(
                step, c["tool"], c["arguments"], output))

        try:
            answer = gemini.generate_answer(rec["question"], results)
        except Exception as error:
            print(f"  skip ({error})")
            continue

        rows.append(to_sft(build_answer_prompt(rec["question"], results), answer))
        print(f"answer {i}/{n}")
        time.sleep(1)  # be gentle with OpenF1 and the Gemini rate limit
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--answers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    checker = F1Agent(model=None, tools={})
    seen = set()
    out = Path(args.out)

    train = generate(TRAIN_TEMPLATES, args.train, rng, checker, seen)
    val = generate(TRAIN_TEMPLATES, args.val, rng, checker, seen)

    # Test = half new questions from training templates, half held-out wording.
    half = args.test // 2
    test = (generate(TRAIN_TEMPLATES, half, rng, checker, seen)
            + generate(HELD_OUT_TEMPLATES, args.test - half, rng, checker, seen))
    for i, rec in enumerate(test):
        rec["held_out_wording"] = i >= half
    rng.shuffle(test)  # so --limit still gets a mix of both kinds

    # Compact JSON = fewer output tokens = faster, cheaper plans.
    def plan_rows(records):
        return [to_sft(build_plan_prompt(r["question"]),
                       json.dumps(r["gold_plan"], ensure_ascii=False))
                for r in records]

    write_jsonl(out / "plan_train.jsonl", plan_rows(train))
    write_jsonl(out / "plan_val.jsonl", plan_rows(val))
    write_jsonl(out / "plan_test.jsonl", test)  # raw: question + gold_plan

    if args.answers:
        write_jsonl(out / "answer_train.jsonl",
                    build_answer_examples(train, args.answers))


if __name__ == "__main__":
    main()
