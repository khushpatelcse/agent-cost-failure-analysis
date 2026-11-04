
from agent import F1Agent, GeminiModel

from tools import (
    get_race_results,
    get_qualifying_results,
    get_fastest_lap,
    get_driver_standings
)

tools = {
    "get_race_results": get_race_results,
    "get_qualifying_results": get_qualifying_results,
    "get_fastest_lap": get_fastest_lap,
    "get_driver_standings": get_driver_standings
}

# Create the Gemini model and the agent.
model = GeminiModel()
agent = F1Agent(model=model, tools=tools)


# Ask the user a question.
question = input("Ask an F1 question: ")

# Run the agent.
result = agent.run(question)

# Show the answer and basic accounting.
print("\nAnswer:")
print(result.answer)

print("\nTool calls:", result.tool_calls_made)
print("Tool failures:", result.tool_failures)
print("Time taken:", result.elapsed_seconds, "seconds")
