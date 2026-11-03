"""
LangChain — Property Assistant

Rebuilds the hand-written RAG pipeline from rag.py with LangChain building blocks:
documents with metadata, a vector store retriever, prompt templates,
a retrieve → prompt → LLM → parse chain, and conversation memory.

Install: pip install langchain-openai langchain-core pandas
Needs the OPENAI_API_KEY environment variable.
"""

import os

import pandas as pd
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)


def describe_listing(row) -> str:
    furnished = "furnished" if row["furnished"] == "Yes" else "unfurnished"
    return (
        f"Listing #{row['property_id']}: {row['bedrooms']} bedroom, "
        f"{row['bathrooms']} bathroom {furnished} {row['property_type']} "
        f"in {row['property_location']}. Price: KES {row['price_kes']:,}."
    )


# ── Building block 2: documents. Each listing becomes a Document with metadata.
# Listings are already short, so they do not need a text splitter;
# long documents such as tenancy agreements would be split into chunks first.
documents = [
    Document(
        page_content=describe_listing(row),
        metadata={
            "property_id": int(row["property_id"]),
            "property_location": row["property_location"],
            "price_kes": int(row["price_kes"]),
            "bedrooms": int(row["bedrooms"]),
            "furnished": row["furnished"],
        },
    )
    for _, row in properties.iterrows()
]
print(f"Created {len(documents)} documents")


# ── Building block 3: a vector store and retriever.
# from_documents() embeds every document in one step, replacing the embed() loop in rag.py.
# Swapping InMemoryVectorStore for Chroma or Pinecone would not change the chain below.
vector_store = InMemoryVectorStore.from_documents(
    documents,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)


def make_retriever(budget=None):
    search_kwargs = {"k": 3}
    if budget is not None:
        # filter receives each Document, so the budget stays an exact rule: price_kes <= budget.
        search_kwargs["filter"] = lambda doc: doc.metadata["price_kes"] <= budget
    return vector_store.as_retriever(search_kwargs=search_kwargs)


# ── Building block 1: a prompt template.
SYSTEM_PROMPT = (
    "You are a Property Assistant. Answer questions ONLY using the provided listings. "
    "Mention listing numbers and prices exactly as given. "
    "If the answer is not in the listings, say 'I don't have that information.'"
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Listings:\n{listings}\n\nQuestion: {question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


# ── Building block 5: a chain. retrieve → format → prompt → LLM → parse.
def make_rag_chain(budget=None):
    retriever = make_retriever(budget)
    # The dictionary runs both branches on the same input:
    # the question goes to the retriever and is also passed through unchanged.
    return (
        {"listings": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )


questions = [
    ("Which furnished apartments are available in Kilimani?", None),
    ("I need a flat in Westlands", 100_000),
    ("Which of these properties allow pets?", None),
]

for question, budget in questions:
    retrieved = make_retriever(budget).invoke(question)
    answer = make_rag_chain(budget).invoke(question)

    print(f"\nQuestion: {question}" + (f" (budget KES {budget:,})" if budget else ""))
    print("Retrieved listings:")
    for doc in retrieved:
        print(f"  {doc.page_content}")
    print("Answer:", answer)


# ── Building block 4: memory. Keep the conversation so follow-up questions make sense.
# MessagesPlaceholder inserts the stored messages into the prompt.
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("user", "Listings:\n{listings}\n\nQuestion: {question}"),
])

chat_chain = chat_prompt | llm | StrOutputParser()
history = []


def chat(question):
    # A follow-up like "Which of those...?" is a poor search on its own,
    # so search with the previous question as well. history[-2] is the last user message.
    search_text = question if not history else f"{history[-2].content} {question}"
    listings = format_docs(make_retriever().invoke(search_text))
    answer = chat_chain.invoke({"history": history, "listings": listings, "question": question})

    # Store the turn so the next question can refer back to it.
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    return answer


print("\n=== Conversation with memory ===")
for question in [
    "Show me apartments in Kilimani.",
    "Which of those is the cheapest?",   # "those" only makes sense with the history
]:
    print(f"User: {question}")
    print(f"Assistant: {chat(question)}\n")

# Compare with rag.py: the steps are the same, but each one is now a swappable piece.
# For this small script, the hand-written version is just as clear;
# the framework pays off when many chains share the same pieces.
