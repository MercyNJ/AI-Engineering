"""
Vector Databases — Property Assistant

Stores the listings from properties.csv in a Chroma collection,
searches them by meaning, and filters them by exact metadata such as price and location.

Install: pip install chromadb pandas
"""

import os

import chromadb
import pandas as pd

# ── Step 1: Turn each property row into a document plus metadata.
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
ids = [f"property_{property_id}" for property_id in properties["property_id"]]

# Metadata is stored next to each document so results can be filtered exactly.
# int() and str() convert pandas values into plain Python types, which Chroma requires.
metadatas = [
    {
        "property_location": str(row.property_location),
        "property_type": str(row.property_type),
        "bedrooms": int(row.bedrooms),
        "bathrooms": int(row.bathrooms),
        "price_kes": int(row.price_kes),
        "furnished": str(row.furnished),
    }
    for row in properties.itertuples()
]


# ── Step 2: Store the listings in a Chroma collection.
# PersistentClient saves the collection to disk, so it is only built once.
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection("property_listings")

# upsert() adds new ids and updates existing ones, so re-running does not create duplicates.
collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
print(f"Collection holds {collection.count()} listings")


# ── Step 3: Search by meaning, with optional exact filters.
def search(query, n_results=3, where=None):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
    )

    print(f"\nQuery: {query}")
    if where:
        print(f"Filter: {where}")

    # distances: lower means more similar.
    for document, distance in zip(results["documents"][0], results["distances"][0]):
        print(f"  {distance:.3f} | {document}")


search("furnished apartment in Kilimani")

# $lte means "less than or equal to": price_kes <= budget.
search("flat in Westlands", where={"price_kes": {"$lte": 100_000}})

# $and combines several conditions; $gte means "greater than or equal to".
search(
    "family home",
    where={
        "$and": [
            {"bedrooms": {"$gte": 3}},
            {"furnished": "Yes"},
        ]
    },
)


# ── Step 4: Garbage in, garbage retrieved.
# An outdated copy of listing #1 with an old price is added by mistake.
collection.upsert(
    documents=["Listing #1: 2 bedroom, 2 bathroom furnished Apartment in Kilimani. Price: KES 55,000."],
    metadatas=[{
        "property_location": "Kilimani",
        "property_type": "Apartment",
        "bedrooms": 2,
        "bathrooms": 2,
        "price_kes": 55_000,
        "furnished": "Yes",
    }],
    ids=["property_1_outdated"],
)

# Chroma returns both versions with confidence: it cannot tell which price is correct.
search("2 bedroom furnished apartment in Kilimani")

# Remove the bad record so only correct listings are indexed.
collection.delete(ids=["property_1_outdated"])
print(f"\nOutdated record deleted. Collection holds {collection.count()} listings")
