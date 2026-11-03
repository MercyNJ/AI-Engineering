"""
Chain-of-Thought & Few-Shot — Making LLMs Smarter

Better results from the same model, with no fine-tuning:
- Chain-of-thought (CoT): make the model reason step by step before answering
- Few-shot CoT: show the reasoning pattern with worked examples
- Self-consistency: sample several answers and take the majority vote

Use: zero-shot CoT for simple to medium reasoning, few-shot CoT for consistent
patterns, self-consistency for maths, logic and high-stakes decisions,
and nothing extra for simple fact lookup or format conversion.

Install: pip install anthropic
Needs the ANTHROPIC_API_KEY environment variable.
"""

import re
from collections import Counter

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"


def call_llm(prompt, max_tokens=1000):
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # content is a list of blocks; pick the text block rather than assuming it comes first.
    return next(block.text for block in response.content if block.type == "text")


# 1. Why standard prompting fails on hard problems: the model pattern-matches the numbers.
standard_prompt = """
If Amara has 3 times as many students as Brian, and together they have
120 students, how many does each have?
"""

# 2. Chain-of-thought: committing to each step constrains the next one.
cot_prompt = """
If Amara has 3 times as many students as Brian, and together they have
120 students, how many does each have?

Let's think step by step:
"""
# Expected reasoning: Brian = x, Amara = 3x, x + 3x = 120 → x = 30.
# Answer: Brian has 30 students, Amara has 90 students.

print("=== Standard prompt ===")
print(call_llm(standard_prompt))
print("\n=== Chain-of-thought prompt ===")
print(call_llm(cot_prompt))


# 3. Zero-shot vs few-shot CoT.
# Zero-shot CoT — just add the magic phrase:
zero_shot = """
Solve this problem. Think step by step.

A course costs KES 15,000. A student pays KES 3,000 per month.
How many months to complete payment?
"""

# Few-shot CoT — show the reasoning pattern with examples:
few_shot = """
Solve each problem by thinking step by step.

Problem: A bus travels 60 km/h for 3 hours. How far does it go?
Solution: Speed × Time = Distance. 60 × 3 = 180 km. Answer: 180 km.

Problem: 5 students share KES 2,500 equally. How much per student?
Solution: Total ÷ Number = Share. 2500 ÷ 5 = 500. Answer: KES 500.

Problem: A course costs KES 15,000. A student pays KES 3,000 per month.
How many months to complete payment?
Solution:"""
# Expected: Total ÷ Monthly = Months. 15000 ÷ 3000 = 5. Answer: 5 months.

print("\n=== Zero-shot CoT ===")
print(call_llm(zero_shot))
print("\n=== Few-shot CoT ===")
print(call_llm(few_shot))


# 4. Self-consistency: sample multiple times, take the majority.
def ask_with_cot(problem, n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"{problem}\n\nThink step by step. Give a final numeric answer."
            }]
        )
        # Extract last number from response
        text = next(block.text for block in response.content if block.type == "text")
        numbers = re.findall(r'\d+', text)
        if numbers:
            answers.append(numbers[-1])

    if not answers:
        return None, {}

    # Counter counts each answer; most_common(1) returns [(answer, count)].
    vote = Counter(answers)
    best = vote.most_common(1)[0][0]
    return best, dict(vote)


answer, votes = ask_with_cot("A train travels at 90km/h. How long to travel 270km?")
print(f"\nAnswer: {answer} hours")
print(f"Vote distribution: {votes}")


# 5. Real production use: a step-by-step loan review.
def evaluate_loan_application(application):
    prompt = f"""
You are a loan risk analyst. Evaluate this application step by step.

Application: {application}

Step 1: Check income vs loan amount ratio
Step 2: Check employment stability
Step 3: Check existing debts
Step 4: Final risk assessment

Give a final verdict: APPROVE, REVIEW, or REJECT with reasoning.
"""
    return call_llm(prompt)


application = (
    "Name: Wanjiru. Monthly income: KES 120,000. Loan requested: KES 500,000 over 24 months. "
    "Employment: 4 years as a nurse at a county hospital. "
    "Existing debts: KES 15,000 per month car loan."
)

print("\n=== Loan Evaluation ===")
print(evaluate_loan_application(application))


# 6. Measure it: standard prompting vs CoT on a test suite with known answers.
def extract_answer(text):
    match = re.search(r"ANSWER:\s*([\d,]+(?:\.\d+)?)", text)
    if not match:
        return None
    # Remove commas so "12,000" is read as 12000, then :g drops a trailing ".0".
    return f"{float(match.group(1).replace(',', '')):g}"


problems = [
    {
        "question": "A course costs KES 12,000. A student pays KES 3,000/month. How many months?",
        "answer":   "4",
    },
    {
        "question": "If a model has 85 correct out of 100 total predictions, what is the total count?",
        "answer":   "100",
    },
    {
        "question": "There are 5 modules with 8 lessons each. What are the total lessons?",
        "answer":   "40",
    },
    {
        "question": "If Amara has 3 times as many students as Brian, and together they have 120 students, how many does Brian have?",
        "answer":   "30",
    },
]

prompt_styles = {
    "Standard Prompting": "Reply with only the final line in the form ANSWER: <number>.",
    "Chain-of-Thought Prompting": "Let's think step by step. End with a final line in the form ANSWER: <number>.",
}

for style, instruction in prompt_styles.items():
    print(f"\n=== {style} ===")
    correct = 0
    for p in problems:
        ans = extract_answer(call_llm(f"{p['question']}\n\n{instruction}"))
        ok  = ans == p["answer"]
        correct += ok
        print(f"  [{'✓' if ok else '✗'}] {p['question'][:50]}... → {ans} (expected {p['answer']})")
    print(f"Accuracy: {correct}/{len(problems)}")
# Easy problems may score the same both ways: CoT matters most on harder, multi-step ones.
