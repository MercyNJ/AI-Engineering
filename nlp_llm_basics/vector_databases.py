"""
Vector Databases — Pinecone, Chroma & FAISS

A vector database stores embeddings and answers one question fast:
"given this new vector, which stored vectors are most similar to it?"
That is the engine behind RAG, semantic search and recommendation systems.

- Chroma:   local and free, no account needed. Best for prototyping.
- Pinecone: managed cloud, scales to millions/billions of vectors. Paid, best for production.
- FAISS:    Meta's free, very fast similarity-search library (not a full database).

Install: pip install chromadb
"""

import chromadb

# Client() keeps the collection in memory, so it disappears when the script ends.
client     = chromadb.Client()
collection = client.create_collection("course_faqs")

# Chroma can embed the text for you automatically
# (the first run downloads a small embedding model).
collection.add(
    documents=[
        "Classes run every Saturday and Sunday, 9am–1pm EAT.",
        "You can pay in monthly instalments of KES 2,500.",
        "Students get lifetime access to all materials after completing the course.",
    ],
    ids=["faq1", "faq2", "faq3"],
)

# query() embeds the question and returns the n_results most similar documents.
results = collection.query(
    query_texts=["When are the live sessions?"],
    n_results=1,
)

# Results are nested lists: one inner list per query text.
print(results["documents"][0][0])
# → "Classes run every Saturday and Sunday, 9am–1pm EAT."

