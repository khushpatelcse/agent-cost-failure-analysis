# model - will be replaced later with our own trained model

import os
import json

class GeminiModel:
    """Uses Gemini to choose tools and write the final answer."""

    def __init__(self, model_name="gemini-3.8-flash"):
        # Import the Gemini library.
        from google import genai
        from google.genai import types

        # Create a connection to Gemini using your environment variable.
        # Get the Gemini API key from the environment.
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
           raise ValueError("GEMINI_API_KEY was not found. Check your .env file.")

        # Connect to Gemini using the API key.
        self.client = genai.Client(api_key=api_key)
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
