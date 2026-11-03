"""
Chain-of-Thought & Few-Shot — Property Assistant

Applies chain-of-thought, few-shot CoT and self-consistency to property questions
from properties.csv, checks every numeric answer against pandas,
and builds a step-by-step property match review for production.

Install: pip install anthropic pandas
Needs the ANTHROPIC_API_KEY environment variable.
"""

import os
import re
from collections import Counter

import anthropic
import pandas as pd

client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"

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


def call_llm(prompt, max_tokens=1000):
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")


# A fixed final line is easier to parse than "the last number in the reply".
ANSWER_FORMAT = "End with a final line in the form ANSWER: <number> (digits only, no commas)."


def extract_answer(text):
    match = re.search(r"ANSWER:\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        number = match.group(1)
    else:
        # Fallback: the last number, keeping commas and decimals so "75,000" is not read as "000".
        numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
        if not numbers:
            return None
        number = numbers[-1]

    # :g drops a trailing ".0", so "91250" and "91250.0" count as the same vote.
    return f"{float(number.replace(',', '')):g}"


# 1. Chain-of-thought on a move-in cost question.
listing = properties.iloc[0]
move_in_problem = (
    f"{listing['description']}\n"
    "The landlord asks for a deposit of two months' rent plus the first month's rent "
    "before moving in. How much does the tenant pay before moving in, in KES?"
)

print("=== Standard prompt ===")
print(call_llm(move_in_problem))
print("\n=== Chain-of-thought prompt ===")
print(call_llm(f"{move_in_problem}\n\nLet's think step by step. {ANSWER_FORMAT}"))

correct_move_in = listing["price_kes"] * 3
print(f"Correct answer from pandas: KES {correct_move_in:,}")


# 2. Few-shot CoT: show the reasoning pattern for monthly costs.
few_shot = f"""
Solve each problem by thinking step by step.

Problem: Rent is KES 60,000 per month and service charge is KES 5,000 per month. What is the yearly cost?
Solution: Monthly total = 60,000 + 5,000 = 65,000. Yearly = 65,000 × 12 = 780,000. ANSWER: 780000

Problem: Two friends share a KES 90,000 apartment equally. How much does each pay per month?
Solution: Share = Total ÷ People = 90,000 ÷ 2 = 45,000. ANSWER: 45000

Problem: {listing['description']} Service charge is KES 7,500 per month. What is the yearly cost?
Solution:"""

print("\n=== Few-shot CoT ===")
print(call_llm(few_shot))
print(f"Correct answer from pandas: KES {(listing['price_kes'] + 7_500) * 12:,}")


# 3. Self-consistency: sample several answers and take the majority vote.
def ask_with_cot(problem, n_samples=5):
    answers = []
    for _ in range(n_samples):
        text = call_llm(f"{problem}\n\nThink step by step. {ANSWER_FORMAT}", max_tokens=1000)
        answer = extract_answer(text)
        if answer:
            answers.append(answer)

    if not answers:
        return None, {}

    vote = Counter(answers)
    best = vote.most_common(1)[0][0]
    return best, dict(vote)


furnished_kilimani = properties[
    (properties["property_location"] == "Kilimani") & (properties["furnished"] == "Yes")
]
average_problem = (
    "Listings:\n" + "\n".join(furnished_kilimani["description"])
    + "\n\nWhat is the average price of these listings, in KES?"
)

answer, votes = ask_with_cot(average_problem)
correct_average = furnished_kilimani["price_kes"].mean()

print("\n=== Self-consistency ===")
print(f"Majority answer: KES {answer}")
print(f"Vote distribution: {votes}")
print(f"Correct answer from pandas: KES {correct_average:,.0f}")
# In production, use pandas for averages like this; self-consistency is for reasoning
# that code cannot easily do, and it costs n_samples times as much.


# 4. Measure it: standard prompting vs CoT on property questions with pandas answers.
first_eight = properties.head(8)
listing_3 = properties.loc[properties["property_id"] == 3].iloc[0]
listing_5 = properties.loc[properties["property_id"] == 5].iloc[0]
listing_2 = properties.loc[properties["property_id"] == 2].iloc[0]

problems = [
    {
        "question": f"{listing_3['description']} Deposit is two months' rent plus the first month. "
                    "How much is paid before moving in, in KES?",
        "answer": listing_3["price_kes"] * 3,
    },
    {
        "question": "Listings:\n" + "\n".join(first_eight["description"])
                    + "\n\nHow many of these listings have a price of at most KES 80,000?",
        "answer": (first_eight["price_kes"] <= 80_000).sum(),
    },
    {
        "question": f"{listing_3['description']}\n{listing_5['description']}\n"
                    "How much more expensive per month is the more expensive listing, in KES?",
        "answer": abs(listing_3["price_kes"] - listing_5["price_kes"]),
    },
    {
        "question": f"{listing_2['description']} What is the rent for a full year, in KES?",
        "answer": listing_2["price_kes"] * 12,
    },
    {
        "question": average_problem,
        "answer": correct_average,
    },
]

prompt_styles = {
    "Standard Prompting": "Reply with only the final line in the form ANSWER: <number> (digits only, no commas).",
    "Chain-of-Thought Prompting": f"Let's think step by step. {ANSWER_FORMAT}",
}

for style, instruction in prompt_styles.items():
    print(f"\n=== {style} ===")
    correct = 0
    for p in problems:
        expected = f"{float(p['answer']):g}"
        ans = extract_answer(call_llm(f"{p['question']}\n\n{instruction}"))
        ok = ans == expected
        correct += ok
        first_line = p["question"].splitlines()[-1]
        print(f"  [{'✓' if ok else '✗'}] {first_line[:55]}... → {ans} (expected {expected})")
    print(f"Accuracy: {correct}/{len(problems)}")


# 5. Real production use: a step-by-step property match review.
def review_property_match(user, listing_description):
    prompt = f"""
You are a Property Assistant reviewing whether a listing suits a user. Check step by step.

User requirements: {user}
Listing: {listing_description}

Step 1: Check location (is the listing in one of the user's preferred locations?)
Step 2: Check price (price_kes <= budget?)
Step 3: Check bedrooms (bedrooms >= minimum_bedrooms?)
Step 4: Check furnishing preference
Step 5: Final assessment

End with a final line in the form VERDICT: MATCH, PARTIAL or NO_MATCH.
MATCH = all checks pass, PARTIAL = only one check fails, NO_MATCH = two or more fail.
"""
    return call_llm(prompt)


def extract_verdict(text):
    match = re.search(r"VERDICT:\s*(MATCH|PARTIAL|NO_MATCH)", text)
    return match.group(1) if match else None


# The same rules in code, so every LLM verdict can be checked.
def rule_based_verdict(user, row):
    failed = 0
    failed += row["property_location"] not in user["locations"]
    failed += row["price_kes"] > user["budget"]
    failed += row["bedrooms"] < user["minimum_bedrooms"]
    failed += (row["furnished"] == "Yes") != user["furnished"]

    if failed == 0:
        return "MATCH"
    return "PARTIAL" if failed == 1 else "NO_MATCH"


user = {
    "locations": ["Kilimani", "Westlands"],
    "budget": 100_000,
    "minimum_bedrooms": 2,
    "furnished": True,
}
user_text = (
    f"preferred locations: {', '.join(user['locations'])}; budget: KES {user['budget']:,}; "
    f"minimum_bedrooms: {user['minimum_bedrooms']}; wants furnished: {user['furnished']}"
)

print("\n=== Property Match Review ===")
print(f"User: {user_text}\n")

agreed = 0
sample = properties.head(6)

for _, row in sample.iterrows():
    review = review_property_match(user_text, row["description"])
    llm_verdict = extract_verdict(review)
    expected = rule_based_verdict(user, row)
    agreed += llm_verdict == expected
    print(f"{row['description']}\n  LLM: {llm_verdict} | Rules: {expected}")

print(f"\nLLM agreed with the rules on {agreed} of {len(sample)} listings")
# Disagreements show where the reasoning went wrong. For hard rules like price_kes <= budget,
# code is the source of truth; the LLM review is useful to explain the result to the user.
