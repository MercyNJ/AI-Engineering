"""
Prompt Patterns That Actually Work

Prompt engineering is structured communication with a model, not magic words.
Covers persona, chain of thought, few-shot examples, output format,
Role + Task + Context + Format, and measuring prompt quality.

Anatomy of a great prompt: Role, Context, Instruction, Format.
Poor: "Tell me about Python."  (no level, goal or format)
Good: "You are a Python tutor. Explain list comprehensions to a beginner
       using one real-world analogy and one code example."

Install: pip install openai
Needs the OPENAI_API_KEY environment variable.
"""

from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY environment variable


# A small helper so each pattern can be sent to the model the same way.
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


# Pattern 1 — Persona: tell the model exactly who to be.
# Weak prompt:
weak_prompt = "Explain REST APIs"

# Strong prompt with persona:
strong_prompt = """
You are a senior software engineer explaining REST APIs to a
junior developer on their first week at a Nairobi fintech startup.
Use simple language, practical examples, and reference M-Pesa or
mobile money APIs as context they'll recognise.

Explain: what is a REST API?
"""

# The persona constrains tone, vocabulary, depth, and relevance.
print("=== Pattern 1: Weak prompt ===")
print(call_llm(weak_prompt))
print("\n=== Pattern 1: Strong prompt with persona ===")
print(call_llm(strong_prompt))


# Pattern 2 — Chain of Thought (CoT): reason step by step before answering.
# Without CoT — often wrong on complex problems:
without_cot = "Is 729 divisible by 9?"

# With CoT:
with_cot = """
Think step by step:
Is 729 divisible by 9?
"""
# → 7+2+9=18, 18÷9=2. Yes, 729 is divisible by 9.

print("\n=== Pattern 2: Without CoT ===")
print(call_llm(without_cot))
print("\n=== Pattern 2: With CoT ===")
print(call_llm(with_cot))


# Pattern 3 — Few-shot examples: show the model the format you expect.
# Three examples is usually enough. More examples = more tokens = more cost.
prompt = """
Classify the sentiment of each review as POSITIVE, NEGATIVE, or NEUTRAL.

Review: "The delivery was super fast!"
Sentiment: POSITIVE

Review: "Product arrived damaged."
Sentiment: NEGATIVE

Review: "It's okay, does what it says."
Sentiment: NEUTRAL

Review: "Absolutely love this! Best purchase this year."
Sentiment:"""

# Model completes: POSITIVE
print("\n=== Pattern 3: Few-shot ===")
print(call_llm(prompt))


# Pattern 4 — Output format specification: say exactly what to return.
# Always specify JSON if you need structured output — otherwise you'll get prose.
prompt = """
Extract the key information from this job posting and return it as JSON:

Job posting:
"We are looking for an AI Engineer at Safaricom. You will build LLM-powered
tools for our M-Pesa platform. Requirements: Python, 3+ years experience,
knowledge of LangChain. Salary: KES 150,000–200,000/month."

Return ONLY valid JSON in this exact format:
{
  "role": "...",
  "company": "...",
  "skills": ["...", "..."],
  "experience_years": ...,
  "salary_range_kes": {"min": ..., "max": ...}
}
"""

import json

print("\n=== Pattern 4: JSON output ===")
response = call_llm(prompt)
print(response)

# json.loads() turns the JSON text into a Python dictionary, or fails if it is not valid JSON.
# Models sometimes wrap JSON in ```json ... ``` fences, so remove them first.
cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()

try:
    job = json.loads(cleaned)
    print("Parsed salary range:", job["salary_range_kes"])
except json.JSONDecodeError:
    print("The model did not return valid JSON.")


# Pattern 5 — Role + Task + Context + Format (+ Constraints).
# ROLE: You are [persona]
# TASK: [what to do]
# CONTEXT: [background information]
# FORMAT: [how to structure the output]
# CONSTRAINTS: [what NOT to do]
system_prompt = """
ROLE: You are a Python code reviewer at a fintech company.
TASK: Review the provided Python function for bugs, security issues, and style.
FORMAT: Return your review as:
  - BUGS: [list any bugs]
  - SECURITY: [list any security concerns]
  - STYLE: [list style improvements]
  - VERDICT: APPROVE or REQUEST_CHANGES
CONSTRAINTS: Be concise. Max 2 sentences per item.
"""

# A small function with deliberate problems for the reviewer to find.
code_to_review = '''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    result = db.execute(query)
    return result[0]
'''

print("\n=== Pattern 5: Role + Task + Context + Format ===")
print(call_llm(code_to_review, system=system_prompt))


# Measuring prompt quality: never judge a prompt on one result.
# Run it on 10+ test cases and measure how often it is right.
def evaluate_prompt(prompt_fn, test_cases, expected_outputs):
    correct = 0
    for inp, expected in zip(test_cases, expected_outputs):
        response = call_llm(prompt_fn(inp))
        if expected.lower() in response.lower():
            correct += 1
    return correct / len(test_cases)


# prompt_fn builds the full prompt for one input.
def my_prompt(review):
    return (
        "Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL. "
        f"Reply with one word.\n\nReview: \"{review}\"\nSentiment:"
    )


test_inputs = [
    "Arrived a day early and works perfectly.",
    "Broke after two uses.",
    "It does the job.",
    "Customer service never replied to my emails.",
    "Great value, I would buy it again.",
    "The colour is slightly different from the photo, but fine.",
    "Worst purchase I have ever made.",
    "Exactly as described.",
    "Packaging was torn and one item was missing.",
    "My kids love it!",
]
expected = [
    "POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE", "POSITIVE",
    "NEUTRAL", "NEGATIVE", "NEUTRAL", "NEGATIVE", "POSITIVE",
]

accuracy = evaluate_prompt(my_prompt, test_inputs, expected)
print(f"\nPrompt accuracy: {accuracy:.0%}")
