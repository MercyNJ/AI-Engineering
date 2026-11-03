"""
How LLMs Work — Property Assistant

Applies tokens, next-token prediction, temperature and context windows
to the listings in properties.csv.

Install: pip install tiktoken numpy pandas
"""

import os

import numpy as np
import pandas as pd
import tiktoken

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)


def describe_listing(row) -> str:
    furnished = "furnished" if row["furnished"] == "Yes" else "unfurnished"
    return (
        f"Listing #{row['property_id']}: {row['bedrooms']} bedroom, "
        f"{row['bathrooms']} bathroom {furnished} {row['property_type']} "
        f"in {row['property_location']}. Price: KES {row['price_kes']:,}."
    )


documents = properties.apply(describe_listing, axis=1).tolist()


# 1. Tokens: what the model actually sees for a listing.
enc = tiktoken.encoding_for_model("gpt-4o")

listing = documents[0]
tokens = enc.encode(listing)

print(f"Listing: {listing}")
print(f"Token IDs: {tokens}")
# decode() on a single ID shows the text piece behind each token.
print(f"Pieces: {[enc.decode([token]) for token in tokens]}")
print(f"Count: {len(tokens)}")

# Users may write in English or Kiswahili. Compare a simple word count with real tokens.
import re


def simple_tokenise(text):
    """Very simplified tokeniser — real ones are more complex"""
    return re.findall(r"\w+|[^\w\s]", text.lower())


requests = [
    "I am looking for a two bedroom house in Kilimani.",
    "Natafuta nyumba ya vyumba viwili Kilimani.",  # Kiswahili: the same request
]

print("\n=== English vs Kiswahili Requests ===")
for request in requests:
    print(f"'{request}'")
    print(f"  Words: {len(simple_tokenise(request))} | Real GPT-4o tokens: {len(enc.encode(request))}")
# Fewer words does not always mean fewer tokens: less common languages split into more pieces.


# 2. Next-token prediction: a request is a prompt the model continues.
prompt = "The best location for a 2 bedroom furnished apartment is"
# Model must predict the next token, e.g. " Kilimani".

# Illustrative logits: score each location by how many matching listings it has.
# np.log() turns counts into scores, like the logits a model would produce.
matching = properties[
    (properties["bedrooms"] == 2)
    & (properties["furnished"] == "Yes")
    & (properties["property_type"] == "Apartment")
]
counts = matching["property_location"].value_counts()

locations = counts.index.tolist()
logits = np.log(counts.values.astype(float))

print(f"\nPrompt: {prompt} ...")
print("Matching listings per location:")
print(counts.to_string())


# 3. Temperature: how predictable the assistant's choice is.
def sample_with_temperature(logits, temperature=1.0):
    if temperature == 0:
        return np.argmax(logits)  # Always pick the top token (deterministic)
    scaled = logits / temperature
    scaled -= scaled.max()  # numerical stability
    probs  = np.exp(scaled) / np.exp(scaled).sum()
    return np.random.choice(len(probs), p=probs)


np.random.seed(42)

# Counter() counts how often each location was picked over 20 samples.
from collections import Counter

for temperature in [0, 1.0, 2.0]:
    picks = Counter(
        locations[sample_with_temperature(logits, temperature)]
        for _ in range(20)
    )
    print(f"\nTemperature {temperature}: {dict(picks)}")

# A Property Assistant giving facts (prices, matches) should use a low temperature;
# a higher one only suits creative text such as listing descriptions.


# 4. Context window: can every listing fit in one prompt?
all_listings_prompt = "Listings:\n" + "\n".join(documents)
prompt_tokens = len(enc.encode(all_listings_prompt))
tokens_per_listing = prompt_tokens / len(documents)

context_windows = {
    "GPT-3.5": 4_096,
    "GPT-4o": 128_000,
    "Claude 3.5": 200_000,
    "Gemini 1.5": 1_000_000,
}

print(f"\nAll {len(documents)} listings: {prompt_tokens:,} tokens "
      f"(about {tokens_per_listing:.0f} per listing)")

for model_name, window in context_windows.items():
    max_listings = int(window // tokens_per_listing)
    print(f"  {model_name:<11} {window:>9,} tokens → room for about {max_listings:,} listings")

# With thousands of listings, even large windows fill up, so RAG retrieves only the relevant ones.


# 5. Token cost: what does one Property Assistant request cost?
# A RAG request sends a system prompt, the top 3 listings and the question.
system_prompt = (
    "You are a Property Assistant. Answer questions ONLY using the provided listings. "
    "If the answer is not in the listings, say 'I don't have that information.'"
)
rag_prompt = (
    system_prompt
    + "\n\nListings:\n" + "\n".join(documents[:3])
    + "\n\nQuestion: Which furnished apartments are available in Kilimani?"
)

input_tokens = len(enc.encode(rag_prompt))
output_tokens = 100   # assumed length of the answer
tokens_per_request = input_tokens + output_tokens

# Illustrative prices per 1,000 tokens, not current ones. Real APIs charge
# input and output tokens separately, and a "free" open model still costs money to host.
models = [
    ("GPT-4o",          0.030),
    ("Claude-3-Sonnet", 0.024),
    ("Gemini-Pro",      0.0035),
    ("Llama-3",         0.000),
]
calls_per_day = 1000

print(f"\nOne RAG request: {input_tokens} input + {output_tokens} output = {tokens_per_request} tokens")
print(f"Sending all {len(documents)} listings instead: {prompt_tokens:,} tokens per request")

print(f"\n=== Cost at {calls_per_day:,} requests per day ===")
for model, price in models:
    rag_daily = (tokens_per_request / 1000) * price * calls_per_day
    all_daily = ((prompt_tokens + output_tokens) / 1000) * price * calls_per_day
    print(f"  {model:<16} RAG: ${rag_daily * 30:8.2f}/month | all listings: ${all_daily * 30:8.2f}/month")


# 6. What LLMs don't know: do the maths in code, not in the model.
# LLMs can make arithmetic errors, so calculate the numbers with pandas
# and give the model the results.
budget = 80_000

within_budget = properties[properties["price_kes"] <= budget]
average_price = properties.groupby("property_location")["price_kes"].mean().round()

print(f"\nListings within a budget of KES {budget:,}: {len(within_budget)} of {len(properties)}")
print("Average price per location (KES):")
print(average_price.astype(int).to_string())

# These listings are also private data: the model never saw them in training,
# so it can only answer about them when they are injected into the prompt (RAG).
