"""
How LLMs Actually Work 

LLMs like GPT-4, Claude and Gemini predict the next token, over and over again.
Covers tokens, next-token prediction, temperature, context windows and token costs.

Install: pip install tiktoken numpy
"""

# 1. What is a token? A piece of text, roughly a word fragment.
# "Hello, how are you?" might tokenise to:
# ["Hello", ",", " how", " are", " you", "?"]

# "artificial intelligence" might be:
# ["art", "if", "icial", " intel", "lig", "ence"]

# LLMs don't see words — they see token IDs (numbers). "Hello" might be token #15339.

# Using tiktoken (OpenAI's tokenizer)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
text = "AI Engineering is about building applications."
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Count: {len(tokens)}")
# Count: ~8 tokens for this sentence


# Tokenisation simulator (no API needed): a simple word-level tokeniser.
import re


def simple_tokenise(text):
    """Very simplified tokeniser — real ones are more complex"""
    # \w+ matches whole words; [^\w\s] matches single punctuation marks.
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens


texts = [
    "Hello, how are you?",
    "AI Engineering is about building applications using pre-trained models.",
    "Ninakupenda Kenya yangu.",  # Kiswahili
    "def calculate_accuracy(correct, total): return correct / total",
]

print("\n=== Tokenisation ===")
for text in texts:
    tokens = simple_tokenise(text)
    print(f"\n'{text[:50]}...' " if len(text) > 50 else f"\n'{text}'")
    print(f"  Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens)>10 else ''}")
    # Compare with the real tokeniser: non-English words usually split into more tokens.
    print(f"  Real GPT-4o tokens: {len(enc.encode(text))}")


# 2. The core task: next-token prediction.
# Training: take huge amounts of text, predict the next token at every position,
# and adjust the weights to make correct predictions more likely. No labels, no rules.

# The model sees this as a sequence of prediction problems:
text = "The capital of Kenya is"
# Model must predict: "Nairobi"

text = "def calculate_accuracy(correct, total):"
# Model must predict: "\n    return correct / total"

# At scale, the model learns grammar, facts, reasoning and code as a side effect.


# 3. Temperature controls creativity.
# The model gives a score (logit) for every possible next token.
# Temperature changes how we sample from those scores.
import numpy as np


def sample_with_temperature(logits, temperature=1.0):
    if temperature == 0:
        return np.argmax(logits)  # Always pick the top token (deterministic)
    # Dividing by a small temperature makes the top token dominate;
    # a large temperature flattens the probabilities.
    scaled = logits / temperature
    probs  = np.exp(scaled) / np.exp(scaled).sum()
    # np.random.choice() picks an index using the probabilities in p.
    return np.random.choice(len(probs), p=probs)


# Assume the model gives these logits for next tokens
# ["Nairobi", "Mombasa", "Lagos", "Accra"]
logits = np.array([5.2, 2.1, 1.8, 1.5])

print("\nTemperature 0.0 (deterministic):")
for _ in range(5): print(["Nairobi","Mombasa","Lagos","Accra"][sample_with_temperature(logits, 0)])

print("Temperature 1.0 (default):")
for _ in range(5): print(["Nairobi","Mombasa","Lagos","Accra"][sample_with_temperature(logits, 1.0)])

print("Temperature 2.0 (creative/random):")
for _ in range(5): print(["Nairobi","Mombasa","Lagos","Accra"][sample_with_temperature(logits, 2.0)])


# Temperature simulation with several samples per temperature.
print("\n=== Temperature Effect ===")


def sample_next_word(candidates, logits, temperature=1.0, n=5):
    if temperature == 0:
        return [candidates[np.argmax(logits)]] * n
    scaled = np.array(logits, dtype=float) / temperature
    # Subtracting the max keeps np.exp() from overflowing; the probabilities stay the same.
    scaled -= scaled.max()  # numerical stability
    probs = np.exp(scaled) / np.exp(scaled).sum()
    return [candidates[np.random.choice(len(candidates), p=probs)] for _ in range(n)]


# Simulated next-word predictions for "The best programming language is"
candidates = ["Python", "JavaScript", "Rust",   "Go",    "Java",   "random_noise"]
logits     = [6.5,      4.2,          3.1,      2.8,     2.5,      0.1]

for temp in [0.0, 0.5, 1.0, 2.0]:
    samples = sample_next_word(candidates, logits, temp)
    print(f"  temp={temp}: {samples}")


# 4. Context window: the model's memory.
# An LLM can only see a limited number of tokens at once:
#   GPT-3.5:    4,096 tokens (~3,000 words)
#   GPT-4o:     128,000 tokens (~96,000 words)
#   Claude 3.5: 200,000 tokens (~150,000 words)
#   Gemini 1.5: 1,000,000 tokens
# Everything outside the context window is forgotten.
# This is why RAG is so powerful: you inject what the model needs to know.


# 5. Token cost calculator: APIs charge per token.
# These prices are illustrative and out of date. Real APIs charge separately
# for input and output tokens, and a "free" open model still costs money to host.
print("\n=== API Cost Estimator ===")
models = [
    ("GPT-4o",         0.030),
    ("Claude-3-Sonnet",0.024),
    ("Gemini-Pro",     0.0035),
    ("Llama-3",        0.000),
]
calls_per_day = 1000
avg_tokens    = 500

# price is per 1,000 tokens, so divide the tokens by 1,000.
for model, price in models:
    daily_cost  = (avg_tokens / 1000) * price * calls_per_day
    monthly_cost= daily_cost * 30
    print(f"  {model:22}: ${daily_cost:6.2f}/day  ${monthly_cost:7.2f}/month")

