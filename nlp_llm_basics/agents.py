"""
AI Agents — Models That Take Actions

An agent is an LLM in a loop: given a goal and a set of tools (functions it can call),
it repeatedly
  1. Think   — decides what information or action it needs next
  2. Act     — calls a tool using structured function calling
  3. Observe — the tool's result is fed back into the conversation
  4. Repeat or finish — stops once it has enough to answer

Install: pip install openai
Needs the OPENAI_API_KEY environment variable.
"""

from openai import OpenAI
client = OpenAI()


# The tool: a normal Python function the model can ask us to run.
def get_course_price(course_name: str) -> str:
    prices = {"ai engineering": "KES 14,999", "python mastery": "KES 999"}
    # get() returns the second argument when the key is missing.
    return prices.get(course_name.lower(), "Course not found")


# The tool description the model sees: its name, purpose and arguments (as JSON Schema).
tools = [{
    "type": "function",
    "function": {
        "name": "get_course_price",
        "description": "Look up the price of an ABC Academy course",
        "parameters": {
            "type": "object",
            "properties": {"course_name": {"type": "string"}},
            "required": ["course_name"],
        },
    },
}]

messages = [{"role": "user", "content": "How much is the AI Engineering course?"}]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)

# The model returns a tool call instead of text. (If it answered directly, tool_calls would be None.)
tool_call = response.choices[0].message.tool_calls[0]

# The model decided to call our function — we run it and hand back the result
import json
# The arguments arrive as a JSON string, e.g. '{"course_name": "AI Engineering"}'.
args   = json.loads(tool_call.function.arguments)
# ** unpacks the dictionary into keyword arguments: get_course_price(course_name="AI Engineering").
result = get_course_price(**args)

# Add the model's tool request and our result, so the model can see both.
messages.append(response.choices[0].message)
messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

final = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
print(final.choices[0].message.content)
# → "The AI Engineering course costs KES 14,999."

# The model never executes your function; it only decides to call it and writes the arguments.
# Your code runs the real function: the model proposes, your code disposes.
#
# Safety basics: agents can loop, call a tool that doesn't exist, or take an irreversible action
# by mistake. Give them only the tools they need, cap the number of loop iterations,
# and require human confirmation before any high-stakes action.
