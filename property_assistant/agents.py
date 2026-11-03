"""
AI Agents — Property Assistant

A Property Assistant agent that searches properties.csv with tools,
runs the full think → act → observe loop, and follows the safety basics:
narrow tools, a step limit, and human confirmation before booking a viewing.

Install: pip install openai pandas
Needs the OPENAI_API_KEY environment variable.
"""

import json
import os

import pandas as pd
from openai import OpenAI

client = OpenAI()

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)

LISTING_COLUMNS = [
    "property_id", "property_location", "property_type",
    "bedrooms", "bathrooms", "price_kes", "furnished",
]


# ── Tools: normal Python functions. The filters are exact, so prices are never guessed.
def search_properties(location=None, max_price=None, min_bedrooms=None, furnished=None):
    matches = properties
    if location:
        matches = matches[matches["property_location"].str.lower() == location.lower()]
    if max_price is not None:
        matches = matches[matches["price_kes"] <= max_price]
    if min_bedrooms is not None:
        matches = matches[matches["bedrooms"] >= min_bedrooms]
    if furnished is not None:
        matches = matches[matches["furnished"] == ("Yes" if furnished else "No")]

    # Sorting by price means "the cheapest" is simply the first result.
    matches = matches.sort_values("price_kes")

    # Return at most 5 listings to keep the context small.
    # to_dict(orient="records") turns each row into a dictionary.
    return json.dumps({
        "total_matches": len(matches),
        "listings": matches[LISTING_COLUMNS].head(5).to_dict(orient="records"),
    }, default=int)


def get_property_details(property_id):
    match = properties[properties["property_id"] == property_id]
    if match.empty:
        return f"Error: no listing with id {property_id}"
    return json.dumps(match[LISTING_COLUMNS].iloc[0].to_dict(), default=int)


# Booking is a real-world action, so a human must approve it first.
def book_viewing(property_id, date):
    if properties[properties["property_id"] == property_id].empty:
        return f"Error: no listing with id {property_id}"

    answer = input(f"\nThe agent wants to book a viewing for listing #{property_id} on {date}. Allow? (y/n) ")
    if answer.strip().lower() != "y":
        return "Booking cancelled: the user did not confirm."
    return f"Viewing booked for listing #{property_id} on {date} (simulated)."


# Maps each tool name the model may request to the real function.
AVAILABLE_TOOLS = {
    "search_properties": search_properties,
    "get_property_details": get_property_details,
    "book_viewing": book_viewing,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": "Search property listings. All filters are optional. Results are sorted by price, cheapest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "e.g. Kilimani, Westlands, Ruiru"},
                    "max_price": {"type": "integer", "description": "Maximum price in KES"},
                    "min_bedrooms": {"type": "integer"},
                    "furnished": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_property_details",
            "description": "Get the full details of one listing by its property_id",
            "parameters": {
                "type": "object",
                "properties": {"property_id": {"type": "integer"}},
                "required": ["property_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_viewing",
            "description": "Book a viewing for a listing. The user is asked to confirm before it is booked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_id": {"type": "integer"},
                    "date": {"type": "string", "description": "Requested viewing day, e.g. Saturday"},
                },
                "required": ["property_id", "date"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a Property Assistant. Use the tools to answer from real listings only. "
    "Never guess prices or invent listings. Quote listing ids and prices exactly as returned."
)

MAX_STEPS = 5   # cap the loop so the agent cannot run forever


# ── The agent loop: think → act → observe → repeat or finish.
def run_agent(goal):
    print(f"\n=== Goal: {goal} ===")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    for step in range(1, MAX_STEPS + 1):
        # Think: the model decides whether to answer or call a tool.
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        message = response.choices[0].message
        messages.append(message)

        # Finish: no tool calls means the model is ready to answer.
        if not message.tool_calls:
            print(f"\nAnswer: {message.content}")
            return

        # Act and observe: run every tool the model asked for and feed back the results.
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"Step {step}: {name}({args})")

            # The model may request a tool that does not exist, so check before running it.
            if name in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS[name](**args)
            else:
                result = f"Error: unknown tool '{name}'"

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    print(f"\nStopped: reached the limit of {MAX_STEPS} steps without a final answer.")


# One tool call is enough here.
run_agent("What 3 bedroom properties do you have in Ruiru?")

# Several steps: search, choose the cheapest, then ask the human before booking.
run_agent(
    "Find me a furnished apartment in Kilimani for at most KES 90,000 "
    "and book a viewing for the cheapest one on Saturday."
)

# The data has no information about pets, so the agent should say it does not know.
run_agent("Which Westlands properties allow pets?")
