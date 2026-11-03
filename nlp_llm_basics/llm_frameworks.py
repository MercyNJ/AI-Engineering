"""
LangChain & LlamaIndex — Orchestrating LLM Apps

Frameworks package the patterns from earlier lessons (prompting, RAG, agents)
into reusable building blocks:
  1. Prompt templates          — prompts with fill-in-the-blank variables
  2. Document loaders/splitters — load files and chunk them for embedding
  3. Retrievers                — one interface over any vector database
  4. Memory                    — conversation history bookkeeping
  5. Chains / agent runners    — connect steps into one reusable pipeline

LlamaIndex leans towards RAG and data-heavy pipelines; LangChain is broader.

Install: pip install langchain-openai langchain-core
Needs the OPENAI_API_KEY environment variable.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# {company} and {question} are filled in when the chain runs.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant for {company}."),
    ("user",   "{question}"),
])

# A template can be filled in on its own to see the messages it will send.
print(prompt.format_messages(company="ABC Academy", question="What is the AI Engineering course about?"))

# The "chain" pipes a prompt template into the model, then parses the output as plain text
# | passes each step's output into the next step.
chain = prompt | llm | StrOutputParser()

answer = chain.invoke({
    "company": "ABC Academy",
    "question": "What is the AI Engineering course about?",
})
print(answer)
