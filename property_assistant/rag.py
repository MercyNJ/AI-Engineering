"""
RAG — Property Assistant

Answers user questions about properties using only the listings in properties.csv.
Each listing is a document: index once, retrieve the closest listings per question,
and let the LLM answer from those listings only.

Install: pip install openai numpy pandas
Needs the OPENAI_API_KEY environment variable.
"""

import os

import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY environment variable

# ── Step 1: Turn each property row into a document.
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
print(f"Loaded {len(documents)} listings")
print("Example document:", documents[0])


# ── Step 2: Embed all listings once, at indexing time.
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


# input can also be a list, so all 50 listings are embedded in one API call.
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
)
doc_embeddings = [np.array(item.embedding) for item in response.data]
print(f"Indexed {len(doc_embeddings)} listings")


# ── Step 3: Retrieve the most relevant listings for a question.
# Embeddings match meaning but are unreliable with numbers,
# so the budget is applied first as an exact filter: price_kes <= budget.
def retrieve(query, k=3, budget=None):
    candidates = range(len(documents))
    if budget is not None:
        candidates = [i for i in candidates if properties["price_kes"].iloc[i] <= budget]

    query_embedding = embed(query)
    # Cosine similarity: higher = more similar
    scored = [
        (
            np.dot(query_embedding, doc_embeddings[i]) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embeddings[i])),
            i,
        )
        for i in candidates
    ]
    # sorted(..., reverse=True) puts the highest similarity first.
    top_k = sorted(scored, reverse=True)[:k]
    return [documents[i] for _, i in top_k]


# ── Step 4: Generate an answer grounded in the retrieved listings.
SYSTEM_PROMPT = (
    "You are a Property Assistant. Answer questions ONLY using the provided listings. "
    "Mention listing numbers and prices exactly as given. "
    "If the answer is not in the listings, say 'I don't have that information.'"
)


def ask(question, budget=None):
    relevant_listings = retrieve(question, k=3, budget=budget)
    context = "\n".join(relevant_listings)

    budget_note = f"\nUser budget: KES {budget:,}" if budget is not None else ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Listings:\n{context}{budget_note}\n\nQuestion: {question}"}
        ]
    )

    print(f"\nQuestion: {question}" + (f" (budget KES {budget:,})" if budget else ""))
    print("Retrieved listings:")
    for listing in relevant_listings:
        print(f"  {listing}")
    print("Answer:", response.choices[0].message.content)


# ── Test it
ask("Which furnished apartments are available in Kilimani?")
ask("I need a flat in Westlands", budget=100_000)
ask("Do you have a family home in Ruiru or Kiambu?")

# Not in the listings, so a grounded answer should say it does not know.
ask("Which of these properties allow pets?")
