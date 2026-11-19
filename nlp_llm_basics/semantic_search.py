import numpy as np


# Measure how similar two vectors are
def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)

    if norm == 0:
        return 0.0

    return np.dot(a, b) / norm


# Create a simple embedding using word counts
def simple_embed(text):
    """Toy embedding for learning how vector similarity works."""
    words = text.lower().split()
    vocab = ["python", "data", "model", "train", "neural", "car", "drive", "road"]

    return np.array([words.count(w) for w in vocab], dtype=float)


# Documents we want to search
docs = [
    "Python is used to train machine learning models",
    "Neural networks need lots of data to train well",
    "The car was driving down the road at high speed",
    "Data scientists build models using Python and R",
]

# User's search query
query = "How do you train a Python model?"

query_vec = simple_embed(query)

# Compare the query with each document
print("BASIC SEMANTIC SEARCH")
print(f"Query: {query}\n")

for doc in docs:
    doc_vec = simple_embed(doc)
    similarity = cosine_similarity(query_vec, doc_vec)

    print(f"[{similarity:.2f}] {doc}")


# --------------------------------------------------
# FAQ search using a simple keyword-based embedding
# --------------------------------------------------

faqs = [
    {
        "q": "How do I reset my password?",
        "a": "Go to the login page and click 'Forgot Password'."
    },
    {
        "q": "What payment methods do you accept?",
        "a": "We accept M-Pesa, card, and bank transfer."
    },
    {
        "q": "How long is the AI course?",
        "a": "The AI Engineering course is 12 weeks long."
    },
    {
        "q": "Do I get a certificate on completion?",
        "a": "Yes! You get a verified certificate after finishing all lessons."
    },
    {
        "q": "Can I pay in installments?",
        "a": "Yes, we offer monthly, weekly and daily installment plans."
    },
    {
        "q": "Is there a mobile app?",
        "a": "The platform is mobile-friendly — no app needed."
    },
]

# Each keyword represents one dimension in the vector
keywords = [
    "password", "login", "payment", "mpesa", "card",
    "course", "weeks", "certificate", "verified",
    "complete", "installment", "monthly", "weekly",
    "mobile", "app"
]


# Convert text into a simple keyword vector
def faq_embed(text):
    text = text.lower()

    return np.array(
        [1.0 if keyword in text else 0.0 for keyword in keywords]
    )


# Find the FAQs that best match a question
def search_faq(query, top_k=2):
    query_vector = faq_embed(query)

    scored_faqs = [
        (
            faq,
            cosine_similarity(
                query_vector,
                faq_embed(faq["q"] + " " + faq["a"])
            )
        )
        for faq in faqs
    ]

    # Put the highest similarity scores first
    scored_faqs.sort(key=lambda x: -x[1])

    return scored_faqs[:top_k]


# Test the FAQ search with different questions
faq_queries = [
    "How do I get my certificate?",
    "What payment options are available?",
    "I forgot my login details",
    "How many weeks is the program?",
]

print("\nFAQ SEARCH")

for query in faq_queries:
    print(f"\nQuery: {query}")

    results = search_faq(query, top_k=2)

    for faq, score in results:
        print(f"[{score:.2f}] Q: {faq['q']}")
        print(f"       A: {faq['a']}")


# --------------------------------------------------
# Real embeddings using OpenAI
# --------------------------------------------------

from openai import OpenAI

client = OpenAI()


# Convert text into a real embedding using OpenAI
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return np.array(response.data[0].embedding)


# Text chunks to search
chunks = [
    "RAG stands for Retrieval-Augmented Generation",
    "The capital city of Kenya is Nairobi",
    "Python lists are created with square brackets",
]

# Convert each chunk into an embedding
chunk_embeddings = [embed(chunk) for chunk in chunks]

# Convert the user's query into an embedding
query = "What does RAG mean in AI?"
query_embedding = embed(query)

# Compare the query with all chunk embeddings
similarities = [
    cosine_similarity(query_embedding, chunk_embedding)
    for chunk_embedding in chunk_embeddings
]

# Find the index of the most similar chunk
best_index = np.argmax(similarities)

print("\nOPENAI EMBEDDING SEARCH")
print(f"Query: {query}")
print(
    f"Best match: {chunks[best_index]} "
    f"(similarity: {similarities[best_index]:.3f})"
)


# --------------------------------------------------
# Build a simple vector store
# --------------------------------------------------

class SimpleVectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = []

    # Store text and its embedding together
    def add(self, text, embedding):
        self.chunks.append(text)
        self.embeddings.append(embedding)

    # Retrieve the most similar text chunks
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
            (self.chunks[index], score)
            for index, score in ranked_results[:top_k]
        ]


# Create an empty vector store
store = SimpleVectorStore()

# Add chunks and their embeddings to the store
for chunk in chunks:
    store.add(chunk, embed(chunk))

# Search the vector store
results = store.search(
    embed("How do I use RAG?"),
    top_k=2
)

print("\nVECTOR STORE SEARCH")

for text, score in results:
    print(f"[{score:.3f}] {text}")