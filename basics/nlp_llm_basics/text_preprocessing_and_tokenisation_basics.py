"""
Text Preprocessing & Tokenisation

Language models convert raw text into numbers before processing it.
Covers text cleaning, tokenisation, BPE, context windows and TF-IDF.

Install: pip install transformers torch tiktoken scikit-learn
"""

# 1. Why preprocessing matters: clean messy raw text.
import re


def clean_text(text: str) -> str:
    """Basic text cleaning pipeline."""
    text = text.lower()                      # normalise case
    text = re.sub(r"<.*?>", " ", text)       # strip HTML tags
    text = re.sub(r"http\S+", " ", text)     # remove URLs
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)  # remove special chars
    text = re.sub(r"\s+", " ", text).strip() # collapse whitespace
    return text


samples = [
    "<p>This is <b>AMAZING</b>! Visit https://example.com for more.</p>",
    "  The model's F1-score was 0.94 on the test set.  ",
]

for s in samples:
    print(f"Raw:     {s}")
    print(f"Cleaned: {clean_text(s)}")
    print()


# 2. Tokenisation: split text into tokens, encode to IDs, decode back.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.tokenize(text)
print(tokens)
# ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

# Encode to input IDs
# return_tensors="pt" returns PyTorch tensors.
encoding = tokenizer(text, return_tensors="pt")
print(encoding["input_ids"])

# Decode back
# BERT adds [CLS] at the start and [SEP] at the end.
decoded = tokenizer.decode(encoding["input_ids"][0])
print(decoded)


# 3. BPE (Byte-Pair Encoding): common words stay whole, rare words split.
from transformers import GPT2Tokenizer

gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")

examples = [
    "Hello world",               # common words
    "tokenization",              # less common
    "supercalifragilistic",      # rare word — will be split
    "AI engineering is fun!",    # a full sentence
]

# In GPT-2 tokens, Ġ marks a token that starts with a space.
for text in examples:
    ids    = gpt2_tok.encode(text)
    pieces = gpt2_tok.convert_ids_to_tokens(ids)
    print(f"'{text}' → {len(ids)} tokens: {pieces}")


# 4. Compare BERT and GPT-2 tokenisation on the same sentences.
# BERT uses WordPiece, where ## marks a piece that continues a word.
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Retrieval-Augmented Generation improves LLM accuracy.",
    "electroencephalography",
]

print("=== BERT Tokenisation ===")
for s in sentences:
    tokens = bert_tok.tokenize(s)
    print(f"  '{s[:40]}...' → {len(tokens)} tokens: {tokens}")

print("\n=== GPT-2 Tokenisation ===")
for s in sentences:
    ids     = gpt2_tok.encode(s)
    pieces  = gpt2_tok.convert_ids_to_tokens(ids)
    print(f"  '{s[:40]}...' → {len(ids)} tokens: {pieces}")


# 5. Context windows and token budgets.
# Context window = the maximum tokens a model can process at once:
#   GPT-2: 1,024 | GPT-4o: 128,000 | Claude 3.5 Sonnet: 200,000 | Gemini 1.5 Pro: 1,000,000
# Rule of thumb: 1 token ≈ 4 characters ≈ 0.75 words in English.
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token per 4 characters."""
    return len(text) // 4


long_doc = "..." * 10000  # simulated long document
tokens_est = estimate_tokens(long_doc)
print(f"Estimated tokens: {tokens_est:,}")

# For accurate counts, use tiktoken (OpenAI's tokeniser)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
exact = len(enc.encode("Your text here"))
print(f"Exact token count: {exact}")


# 6. TF-IDF: classical text representation for keyword search and simple classifiers.
# TF rewards words that appear often in a document.
# IDF penalises words that appear in every document.
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "machine learning is great",
    "deep learning uses neural networks",
    "machine learning and deep learning are both AI",
    "natural language processing uses deep learning",
]

# max_features=20 keeps at most the 20 most frequent words.
vectorizer = TfidfVectorizer(max_features=20)
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
# The corpus only has 15 unique words, so there are 15 features, not 20.
print("Shape:", X.shape)  # (4, 15) — 4 docs, 15 features
