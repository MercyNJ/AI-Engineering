"""
RAG — Give Your AI a Memory

RAG (Retrieval-Augmented Generation) gives an LLM access to a library of documents
at query time. The system finds the most relevant chunks, adds them to the prompt,
and the LLM answers from that context, not just its training data.

INDEXING (done once):   Documents → Chunk + Embed → Vector DB
QUERY (every request):  User question → Embed query → Search vector DB → Top chunks
                        → LLM (prompt = question + top chunks) → Grounded answer

Install: pip install openai numpy
Needs the OPENAI_API_KEY environment variable.
"""

from openai import OpenAI
import numpy as np

client = OpenAI()  # uses OPENAI_API_KEY environment variable

# ── Step 1: Your document library (in production, these come from files/database)
documents = [
    "ABC Academy offers a 12-week AI Engineering programme for KES 14,999.",
    "Students can pay in monthly instalments of KES 2,500 per month.",
    "The course includes Python, ML, Deep Learning, LLMs, and deployment.",
    "Classes run every Saturday and Sunday, 9am–1pm EAT.",
    "Students get lifetime access to all course materials after completion.",
]


# ── Step 2: Embed all documents (once, at indexing time)
# An embedding model turns text into a vector; similar meanings get similar vectors.
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


doc_embeddings = [embed(doc) for doc in documents]
print(f"Indexed {len(documents)} documents")


# ── Step 3: Retrieve the most relevant chunks for a query
def retrieve(query, k=2):
    query_embedding = embed(query)
    # Cosine similarity: higher = more similar
    similarities = [
        np.dot(query_embedding, doc_emb) /
        (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embeddings
    ]
    # Get top-k indices
    # argsort() sorts from lowest to highest, so [::-1] puts the most similar first.
    top_k = np.argsort(similarities)[::-1][:k]
    return [documents[i] for i in top_k]


# ── Step 4: Generate a grounded answer
def ask(question):
    relevant_chunks = retrieve(question, k=2)
    context = "\n".join(relevant_chunks)

    # The system message sets the rules; the user message carries the context and question.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for ABC Academy. Answer questions ONLY using the provided context. If the answer is not in the context, say 'I don't have that information.'"},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content


# ── Test it
print(ask("How much does the course cost?"))
# → "The AI Engineering programme costs KES 14,999. You can also pay in monthly instalments of KES 2,500."
print(ask("Is there a certificate?"))
# → "I don't have that information."  (grounded — no hallucination!)

# This example uses numpy for similarity search. With thousands of documents you would use
# a vector database like Chroma (free, local) or Pinecone (cloud). Only the storage layer changes.
