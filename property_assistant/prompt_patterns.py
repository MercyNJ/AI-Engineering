"""
Prompt Patterns — Property Assistant

Applies persona, chain of thought, few-shot, JSON output and
Role + Task + Context + Format prompts to the listings in properties.csv,
then measures two prompts against labelled test messages.

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


def describe_listing(row) -> str:
    furnished = "furnished" if row["furnished"] == "Yes" else "unfurnished"
    return (
        f"Listing #{row['property_id']}: {row['bedrooms']} bedroom, "
        f"{row['bathrooms']} bathroom {furnished} {row['property_type']} "
        f"in {row['property_location']}. Price: KES {row['price_kes']:,}."
    )


properties["description"] = properties.apply(describe_listing, axis=1)


def call_llm(prompt, system=None, temperature=0):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


# Pattern 1 — Persona: explain a listing to a first-time renter.
listing = properties["description"].iloc[0]

persona_prompt = f"""
You are an experienced Nairobi letting agent speaking to someone renting
for the first time. Use simple language and mention what they should check
during a viewing.

Explain this listing: {listing}
"""

print("=== Pattern 1: Persona ===")
print(call_llm(persona_prompt))


# Pattern 2 — Chain of Thought: check several requirements against several listings.
requirements = "at least 2 bedrooms, furnished, price at most KES 90,000"
candidates = properties.head(8)
listings_text = "\n".join(candidates["description"])

cot_prompt = f"""
Requirements: {requirements}

Listings:
{listings_text}

Think step by step: check each listing against every requirement,
then list the ids of the listings that meet all of them.
"""

print("\n=== Pattern 2: Chain of Thought ===")
print(call_llm(cot_prompt))

# Step-by-step reasoning helps, but the exact answer comes from pandas, so check it.
correct = candidates[
    (candidates["bedrooms"] >= 2)
    & (candidates["furnished"] == "Yes")
    & (candidates["price_kes"] <= 90_000)
]["property_id"].tolist()
print(f"Correct answer from pandas: {correct}")


# Pattern 3 — Few-shot: classify what the user wants.
def few_shot_intent_prompt(message):
    return f"""
Classify each message as SEARCH, VIEWING, COMPLAINT, or PRICE.

Message: "Do you have any 3 bedroom houses in Ruiru?"
Intent: SEARCH

Message: "Can I come and see listing #12 on Friday?"
Intent: VIEWING

Message: "The caretaker was rude and the flat was dirty."
Intent: COMPLAINT

Message: "{message}"
Intent:"""


print("\n=== Pattern 3: Few-shot ===")
print(call_llm(few_shot_intent_prompt("Is the rent for the Westlands maisonette negotiable?")))


# Pattern 4 — Output format: turn a request into JSON filters, then search the CSV.
request = "Looking for a furnished 2 bedroom apartment in Kilimani, my budget is KES 90,000."

json_prompt = f"""
Extract the search filters from this property request and return them as JSON:

Request: "{request}"

Return ONLY valid JSON in this exact format (use null when not mentioned):
{{
  "location": "...",
  "property_type": "...",
  "min_bedrooms": ...,
  "max_price_kes": ...,
  "furnished": true or false
}}
"""
# {{ and }} print literal braces inside an f-string.

print("\n=== Pattern 4: JSON output ===")
response = call_llm(json_prompt)
print(response)

# Models sometimes wrap JSON in ```json ... ``` fences, so remove them first.
cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()

try:
    filters = json.loads(cleaned)
except json.JSONDecodeError:
    filters = None
    print("The model did not return valid JSON.")

if filters:
    matches = properties
    if filters.get("location"):
        matches = matches[matches["property_location"].str.lower() == filters["location"].lower()]
    if filters.get("property_type"):
        matches = matches[matches["property_type"].str.lower() == filters["property_type"].lower()]
    if filters.get("min_bedrooms"):
        matches = matches[matches["bedrooms"] >= filters["min_bedrooms"]]
    if filters.get("max_price_kes"):
        matches = matches[matches["price_kes"] <= filters["max_price_kes"]]
    if filters.get("furnished") is not None:
        matches = matches[matches["furnished"] == ("Yes" if filters["furnished"] else "No")]

    print(f"{len(matches)} matching listings:")
    for description in matches["description"]:
        print(f"  {description}")


# Pattern 5 — Role + Task + Context + Format + Constraints: write a listing advert.
advert_system_prompt = """
ROLE: You are a property copywriter for a Nairobi letting agency.
TASK: Write a short advert for the listing provided.
CONTEXT: Readers are browsing on their phones and compare many listings quickly.
FORMAT:
  - HEADLINE: one line
  - HIGHLIGHTS: 3 bullet points
  - PRICE: the exact price from the listing
CONSTRAINTS: Use only facts from the listing. Do not invent features
(no pools, views or parking unless stated). Do not change the price.
"""

print("\n=== Pattern 5: Role + Task + Context + Format ===")
print(call_llm(listing, system=advert_system_prompt))


# Measuring prompt quality: compare a zero-shot prompt with the few-shot prompt.
def evaluate_prompt(prompt_fn, test_cases, expected_outputs):
    correct = 0
    for inp, expected in zip(test_cases, expected_outputs):
        response = call_llm(prompt_fn(inp))
        if expected.lower() in response.lower():
            correct += 1
    return correct / len(test_cases)


def zero_shot_intent_prompt(message):
    return f"What does the user want? Message: \"{message}\""


test_messages = [
    "Any 2 bedroom apartments in Kileleshwa?",
    "I'd like to view the Ruaka maisonette this weekend.",
    "Nobody fixed the leaking tap I reported last month.",
    "How much is listing #7?",
    "Show me furnished places in Westlands.",
    "Is Saturday morning free for a viewing of #23?",
    "The photos were misleading, the house is much smaller.",
    "What's the rent for the Langata house?",
    "Do you have anything in South B under KES 60,000?",
    "Can I bring my partner to the viewing on Monday?",
]
expected_intents = [
    "SEARCH", "VIEWING", "COMPLAINT", "PRICE", "SEARCH",
    "VIEWING", "COMPLAINT", "PRICE", "SEARCH", "VIEWING",
]

zero_shot_accuracy = evaluate_prompt(zero_shot_intent_prompt, test_messages, expected_intents)
few_shot_accuracy = evaluate_prompt(few_shot_intent_prompt, test_messages, expected_intents)

print("\n=== Prompt Quality ===")
print(f"Zero-shot prompt accuracy: {zero_shot_accuracy:.0%}")
print(f"Few-shot prompt accuracy:  {few_shot_accuracy:.0%}")
# The zero-shot prompt never names the labels, so its replies rarely contain them.
# Showing the format with examples is what makes the output measurable.
