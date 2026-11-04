# agent

"""
F1 Statistics Agent

This file connects our language model to our F1 data tools.

The model decides which tools to use.
The agent checks the plan, runs the tools, collects the results,
and asks the model to write the final answer.
"""

import json
import time
from datetime import datetime


"""Tools the agents can use."""
TOOL_ARGUMENTS = {
    "get_race_results": {
        "year": int,
        "race": str
    },
    "get_qualifying_results": {
        "year": int,
        "race": str
    },
    "get_fastest_lap": {
        "year": int,
        "race": str
    },
    "get_driver_standings": {
        "year": int
    }
}

FIRST_SUPPORTED_YEAR = 2023
LAST_SUPPORTED_YEAR = 2026
MAX_TOOL_CALLS = 4


class AgentError(Exception):
    """An error caused by an invalid plan from the model."""
    pass


class AgentResponse:
    """Stores the answer and information about the agent's run."""

    def __init__(
        self,
        answer,
        tool_calls_made,
        tool_failures,
        elapsed_seconds,
        tool_results
    ):
        self.answer = answer
        self.tool_calls_made = tool_calls_made
        self.tool_failures = tool_failures
        self.elapsed_seconds = elapsed_seconds
        self.tool_results = tool_results


class F1Agent:

    def __init__(self, model, tools, max_tool_calls=MAX_TOOL_CALLS):
        # The model chooses tools and generates answers.
        self.model = model

        # This dictionary connects tool names to Python functions.
        self.tools = tools

        # Limit how many tools can be called for one question.
        self.max_tool_calls = max_tool_calls

    def run(self, question):
        """Handle a user's question from beginning to end."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please enter a question.")

        start_time = time.perf_counter()

        # 1. Ask the model which tools it wants to use.
        raw_plan = self.model.plan_tools(question)

        # 2. Check that the model returned a valid plan.
        plan = self._read_plan(raw_plan)
        tool_calls = plan["tool_calls"]

        if not tool_calls:
            raise AgentError("The model did not choose any tools.")

        if len(tool_calls) > self.max_tool_calls:
            raise AgentError("The model requested too many tool calls.")

        results = []
        failures = 0

        # 3. Run each requested tool.
        for step_number, call in enumerate(tool_calls, start=1):

            tool_name = call["tool"]
            arguments = call["arguments"]

            # Check the tool name and its arguments.
            self._check_tool_call(tool_name, arguments)

            # Check that the tool has actually been connected.
            if tool_name not in self.tools:
                failures += 1

                results.append({
                    "step": step_number,
                    "tool": tool_name,
                    "arguments": arguments,
                    "ok": False,
                    "data": None,
                    "error": "This tool has not been connected yet."
                })

                continue

            try:
                # Run the selected Python tool.
                tool_output = self.tools[tool_name](**arguments)

                # Store the tool result in a consistent format.
                result = self._format_tool_result(
                    step_number,
                    tool_name,
                    arguments,
                    tool_output
                )

                if not result["ok"]:
                    failures += 1

                results.append(result)

            except Exception as error:
                # Record a tool failure instead of crashing immediately.
                failures += 1

                results.append({
                    "step": step_number,
                    "tool": tool_name,
                    "arguments": arguments,
                    "ok": False,
                    "data": None,
                    "error": str(error)
                })

        # 4. Ask the model to answer using the tool results.
        
        try:
            answer = self.model.generate_answer(
                question,
                results
            )

        except Exception as error:
            # Show the real reason Gemini failed.
            print("\nGemini final-answer error:")
            print(type(error).__name__)
            print(error)

            answer = (
                "I collected the tool results, but I couldn't "
                "generate a final answer."
            )

        # 5. Measure how long the complete process took.
        elapsed_time = time.perf_counter() - start_time

        # 6. Return the answer and accounting information.
        return AgentResponse(
            answer=answer,
            tool_calls_made=len(tool_calls),
            tool_failures=failures,
            elapsed_seconds=round(elapsed_time, 4),
            tool_results=results
        )

    def _read_plan(self, raw_plan):
        """Check that the model returned a properly formatted plan."""

        # Sometimes the model returns JSON as text.
        if isinstance(raw_plan, str):
            try:
                raw_plan = json.loads(raw_plan)
            except json.JSONDecodeError as error:
                raise AgentError(
                    "The model did not return valid JSON."
                ) from error

        if not isinstance(raw_plan, dict):
            raise AgentError("The plan must be a JSON object.")

        if set(raw_plan.keys()) != {"tool_calls"}:
            raise AgentError(
                "The plan must contain a 'tool_calls' field."
            )

        tool_calls = raw_plan["tool_calls"]

        if not isinstance(tool_calls, list):
            raise AgentError("'tool_calls' must be a list.")

        # Check that every tool call has the expected structure.
        for call in tool_calls:

            if not isinstance(call, dict):
                raise AgentError("Each tool call must be an object.")

            if set(call.keys()) != {"tool", "arguments"}:
                raise AgentError(
                    "Each tool call needs a tool name and arguments."
                )

            if not isinstance(call["tool"], str):
                raise AgentError("The tool name must be text.")

            if not isinstance(call["arguments"], dict):
                raise AgentError("Tool arguments must be an object.")

        return raw_plan

    def _check_tool_call(self, tool_name, arguments):
        """Make sure the requested tool and arguments are allowed."""

        if tool_name not in TOOL_ARGUMENTS:
            raise AgentError(
                f"The model requested an unknown tool: {tool_name}"
            )

        required_arguments = TOOL_ARGUMENTS[tool_name]

        missing = set(required_arguments) - set(arguments)
        extra = set(arguments) - set(required_arguments)

        if missing:
            raise AgentError(
                f"Missing arguments for {tool_name}: {missing}"
            )

        if extra:
            raise AgentError(
                f"Unexpected arguments for {tool_name}: {extra}"
            )

        # Check that every argument has the correct type.
        for name, expected_type in required_arguments.items():

            value = arguments[name]

            if expected_type is int:
                # Prevent True or False from being accepted as a year.
                correct_type = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                )
            else:
                correct_type = isinstance(value, expected_type)

            if not correct_type:
                raise AgentError(
                    f"{name} must be a {expected_type.__name__}."
                )

            if name == "year":
                if not FIRST_SUPPORTED_YEAR <= value <= LAST_SUPPORTED_YEAR:
                    raise AgentError(
                        f"Year must be between {FIRST_SUPPORTED_YEAR} "
                        f"and {LAST_SUPPORTED_YEAR}."
                    )

            if isinstance(value, str) and not value.strip():
                raise AgentError(f"{name} cannot be empty.")

    @staticmethod
    def _format_tool_result(
        step_number,
        tool_name,
        arguments,
        tool_output
    ):
        """Put tool results into a format the model can understand."""

        # Handle tools that return {"ok": ..., "data": ..., "error": ...}.
        if isinstance(tool_output, dict) and "ok" in tool_output:

            succeeded = tool_output["ok"] is True

            return {
                "step": step_number,
                "tool": tool_name,
                "arguments": arguments,
                "ok": succeeded,
                "data": tool_output.get("data"),
                "error": (
                    None if succeeded
                    else tool_output.get(
                        "error",
                        "The tool reported an error."
                    )
                )
            }

        # Otherwise, assume the tool returned its data directly.
        return {
            "step": step_number,
            "tool": tool_name,
            "arguments": arguments,
            "ok": True,
            "data": tool_output,
            "error": None
        }
    
    """ gemini api until we train our own llm"""


class GeminiModel:
    """Uses Gemini to choose tools and write the final answer."""

    def __init__(self, model_name="gemini-3.8-flash"):
        # Import the Gemini library.
        from google import genai
        from google.genai import types

        # Create a connection to Gemini using your environment variable.
        self.client = genai.Client()
        self.types = types
        self.model_name = model_name

    def plan_tools(self, question):
        """Ask Gemini which F1 tools should be called."""

        prompt = f"""
You are the planning model for an F1 statistics agent.

Choose the tool or tools needed to answer the user's question.

Available tools:

1. get_race_results
   Arguments: year (integer), race (string)
   Use for race results, finishing positions, and race winners.

2. get_qualifying_results
   Arguments: year (integer), race (string)
   Use for qualifying results and pole position.

3. get_fastest_lap
   Arguments: year (integer), race (string)
   Use for the fastest lap during a race.

4. get_driver_standings
   Arguments: year (integer)
   Use for the championship standings for a season.

Return only a JSON object with this format:

{{
  "tool_calls": [
    {{
      "tool": "get_race_results",
      "arguments": {{
        "year": 2025,
        "race": "Miami"
      }}
    }}
  ]
}}

For questions requiring multiple tools, include multiple
items in tool_calls.

Do not answer the F1 question yourself.
Only create the tool plan.

User question: {question}
"""

        # Tell Gemini the structure its response must follow.
        
        plan_schema = {
            "type": "OBJECT",
            "properties": {
                "tool_calls": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "tool": {
                                "type": "STRING",
                                "enum": [
                                    "get_race_results",
                                    "get_qualifying_results",
                                    "get_fastest_lap",
                                    "get_driver_standings"
                                ]
                            },
                            "arguments": {
                                "type": "OBJECT",
                                "properties": {
                                    "year": {
                                        "type": "INTEGER"
                                    },
                                    "race": {
                                        "type": "STRING"
                                    }
                                }
                            }
                        },
                        "required": ["tool", "arguments"]
                    }
                }
            },
            "required": ["tool_calls"]
        }

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=plan_schema
            )
        )

        if not response.text:
            raise ValueError("Gemini returned an empty tool plan.")

        # The agent will validate this JSON before executing anything.
        return response.text

    def generate_answer(self, question, tool_results):
        """Ask Gemini to answer using the tool results."""

        prompt = f"""
You are an assistant answering Formula 1 statistics questions.

Original user question:
{question}

Results returned by the tools:
{json.dumps(tool_results, indent=2, ensure_ascii=False, default=str)}

Instructions:
- Answer the user's original question clearly.
- Base factual claims on the returned tool data.
- Do not invent missing statistics.
- If a tool failed or returned no data, explain that honestly.
- For a comparison, use the results from all relevant tools.
- Keep the answer concise and easy to understand.
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        if not response.text:
            return "I couldn't generate an answer from the tool results."

        return response.text
