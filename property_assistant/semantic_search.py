import numpy as np
from openai import OpenAI

client = OpenAI()


# Convert text into an embedding vector
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return np.array(response.data[0].embedding)


# Measure how similar two vectors are
def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)

    if norm == 0:
        return 0.0

    return np.dot(a, b) / norm


# Store property descriptions and their embeddings
class SimpleVectorStore:
    def __init__(self):
        self.properties = []
        self.embeddings = []

    # Add a property and its embedding
    def add(self, property_description, embedding):
        self.properties.append(property_description)
        self.embeddings.append(embedding)

    # Search for properties matching the user's query
    def search(self, query_embedding, top_k=3):
        similarities = [
            cosine_similarity(query_embedding, embedding)
            for embedding in self.embeddings
        ]

        ranked_results = sorted(
            enumerate(similarities),
            key=lambda x: -x[1]
        )

        return [
            (self.properties[index], score)
            for index, score in ranked_results[:top_k]
        ]


# Sample property descriptions
properties = [
    "Two-bedroom apartment in Ruiru with parking, borehole water and balcony.",
    "One-bedroom furnished apartment in Kilimani with Wi-Fi and swimming pool.",
    "Three-bedroom house in Karen with garden, garage and backup water storage.",
    "Two-bedroom apartment in Roysambu with secure parking and rooftop access.",
    "Studio apartment in Westlands with gym, furnished interior and high-speed internet."
]


# Create an empty property vector store
property_store = SimpleVectorStore()


# Add each property and its embedding to the store
for property_description in properties:
    property_store.add(
        property_description,
        embed(property_description)
    )


# Test different property search queries
queries = [
    "I need a home with reliable water and somewhere to park my car.",
    "I am looking for a furnished apartment.",
    "I want a house with a garden.",
    "I need somewhere with a swimming pool and gym."
]


# Search for properties matching each query
for query in queries:
    query_embedding = embed(query)

    results = property_store.search(
        query_embedding,
        top_k=2
    )

    print(f"\nSearch query: {query}")
    print("Most relevant properties:\n")

    for property_description, score in results:
        print(f"[{score:.3f}] {property_description}")