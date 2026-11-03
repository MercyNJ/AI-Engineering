"""
Text Preprocessing & Tokenisation

Applies text cleaning, tokenisation, token budgets and TF-IDF
to property listings built from properties.csv and to user requests.

Install: pip install transformers tiktoken scikit-learn pandas
"""

# 1. Load the properties and turn each row into a text listing.
import os
import re

import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)


def describe_listing(row) -> str:
    furnished = "furnished" if row["furnished"] == "Yes" else "unfurnished"
    return (
        f"{row['bedrooms']} bedroom, {row['bathrooms']} bathroom "
        f"{furnished} {row['property_type']} in {row['property_location']}. "
        f"Price: KES {row['price_kes']:,}."
    )


# apply(axis=1) runs the function once per row.
properties["description"] = properties.apply(describe_listing, axis=1)

print(f"Dataset: {len(properties)} properties")
print(properties[["property_id", "description"]].head())


# 2. Clean messy user requests before using them.
def clean_text(text: str) -> str:
    """Basic text cleaning pipeline."""
    text = text.lower()                      # normalise case
    text = re.sub(r"<.*?>", " ", text)       # strip HTML tags
    text = re.sub(r"http\S+", " ", text)     # remove URLs
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)  # remove special chars
    text = re.sub(r"\s+", " ", text).strip() # collapse whitespace
    return text


user_requests = [
    "<p>Looking for a <b>2 BEDROOM</b> Apartment in Kilimani!!! Saw it on https://example.com/listing/12</p>",
    "   Need a FURNISHED house in Ruaka :) max budget KES 60,000   ",
    "3-bedroom Maisonette in Westlands?? #urgent",
]

print("\n=== Cleaned User Requests ===")
for request in user_requests:
    print(f"Raw:     {request}")
    print(f"Cleaned: {clean_text(request)}")
    print()

cleaned_requests = [clean_text(request) for request in user_requests]
properties["clean_description"] = properties["description"].apply(clean_text)


# 3. Tokenise a listing with BERT (WordPiece) and GPT-2 (BPE).
from transformers import AutoTokenizer

bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

listing = properties["description"].iloc[0]

print("=== Listing Tokenisation ===")
print(f"Listing: {listing}")
print(f"  BERT  → {bert_tok.tokenize(listing)}")
print(f"  GPT-2 → {gpt2_tok.convert_ids_to_tokens(gpt2_tok.encode(listing))}")

# Local place names are rare in the training text, so they split into subwords.
print("\n=== Location Tokenisation ===")
for location in sorted(properties["property_location"].unique()):
    bert_pieces = bert_tok.tokenize(location)
    gpt2_pieces = gpt2_tok.convert_ids_to_tokens(gpt2_tok.encode(location))
    print(f"  {location:<11} BERT: {bert_pieces} | GPT-2: {gpt2_pieces}")


# 4. Token budget: will all the listings fit in one prompt?
import tiktoken


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token per 4 characters."""
    return len(text) // 4


prompt = (
    "You are a property assistant. Recommend listings that match the request.\n\n"
    "Listings:\n"
    + "\n".join(
        f"{row.property_id}. {row.description}"
        for row in properties.itertuples()
    )
    + f"\n\nRequest: {cleaned_requests[0]}"
)

enc = tiktoken.encoding_for_model("gpt-4o")
exact_tokens = len(enc.encode(prompt))
gpt2_tokens = len(gpt2_tok.encode(prompt))

print("\n=== Prompt Token Budget ===")
print(f"Characters:        {len(prompt):,}")
print(f"Estimated tokens:  {estimate_tokens(prompt):,}")
print(f"GPT-4o tokens:     {exact_tokens:,}")
print(f"GPT-2 tokens:      {gpt2_tokens:,}")

context_windows = {
    "GPT-2": 1_024,
    "GPT-4o": 128_000,
}

tokens_per_listing = exact_tokens / len(properties)

for model_name, window in context_windows.items():
    fits = "fits" if exact_tokens <= window else "does NOT fit"
    max_listings = int(window // tokens_per_listing)
    print(f"{model_name} ({window:,} tokens): prompt {fits}, room for about {max_listings:,} listings")


# 5. TF-IDF keyword search: match each request to the closest listings.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
listing_vectors = vectorizer.fit_transform(properties["clean_description"])

print("\n=== TF-IDF Listing Vectors ===")
print("Vocabulary:", vectorizer.get_feature_names_out())
print("Shape:", listing_vectors.shape)

# The default token pattern ignores single characters like "2",
# so TF-IDF matches words such as locations and property types, not bedroom counts.
# It also matches numbers as plain words: "60" in a budget matches listings priced KES 60,000,
# even in a different location. TF-IDF matches keywords, not meaning.
request_vectors = vectorizer.transform(cleaned_requests)

# cosine_similarity() scores how similar each request is to each listing (0 to 1).
similarities = cosine_similarity(request_vectors, listing_vectors)

print("\n=== Top 3 Listings per Request ===")
for request, scores in zip(cleaned_requests, similarities):
    print(f"Request: {request}")

    # argsort() sorts from lowest to highest, so [::-1] puts the best first.
    top_indices = scores.argsort()[::-1][:3]

    for index in top_indices:
        row = properties.iloc[index]
        print(f"  {scores[index]:.2f} | #{row['property_id']} {row['description']}")
    print()
