"""
Transformers & BERT Explained

The Transformer ("Attention Is All You Need", 2017) is the architecture behind
GPT-4, Claude, Gemini and most modern language models.
Covers self-attention, multi-head attention, the transformer block,
BERT (encoder-only), GPT (decoder-only) and fine-tuning BERT.

Install: pip install transformers torch datasets accelerate
"""

# 1. Self-attention: every token looks at every other token at the same time.
# RNNs/LSTMs process one step at a time, so they cannot parallelise
# and early tokens fade over long sequences. Self-attention solves both.
import numpy as np


def self_attention(Q, K, V, scale=True):
    """
    Compute scaled dot-product attention.

    Q — Query matrix: what this token is looking for
    K — Key matrix:   what each token represents as searchable info
    V — Value matrix: what information each token will contribute
    All have shape (seq_len, d_k)
    """
    d_k   = Q.shape[-1]
    # Step 1: compute similarity between every query and every key
    # @ is matrix multiplication and .T is the transpose.
    scores = Q @ K.T                         # (seq_len, seq_len)
    if scale:
        scores = scores / np.sqrt(d_k)       # prevents softmax saturation

    # Step 2: softmax → attention weights (each row sums to 1)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    # Step 3: weighted sum of values
    output = weights @ V                     # (seq_len, d_model)
    return output, weights


# Small demonstration
np.random.seed(42)
seq_len, d_k = 4, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output, attn_weights = self_attention(Q, K, V)
print("Attention weights (each row sums to 1):")
print(attn_weights.round(3))
print("Output shape:", output.shape)


# 2. Multi-head attention and the full transformer block.
# Each head learns to attend to different aspects (syntax, coreference, semantics):
#   MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W_o
#   where headᵢ = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)
#
# Transformer block:
#   Input → Positional Encoding
#     → Multi-Head Self-Attention → + residual → Layer Normalisation
#     → Feed-Forward Network (2 dense layers) → + residual → Layer Normalisation
#     → Output
#
# Positional encoding adds token order, because attention alone ignores order.
# Residual connections let gradients flow freely, so transformers can be very deep.


# 3. BERT: an encoder-only transformer that reads the whole input at once (bidirectional).
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")

text     = "AI engineering is transforming the tech industry."
encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

# no_grad() turns off gradient tracking, which saves memory when only predicting.
with torch.no_grad():
    # ** unpacks the dictionary into input_ids=..., attention_mask=..., etc.
    outputs = model(**encoding)

# outputs.last_hidden_state — shape (batch, seq_len, 768)
# Each token has a 768-dimensional contextual representation
print("\nHidden states shape:", outputs.last_hidden_state.shape)

# [CLS] token embedding — used for classification tasks
cls_embedding = outputs.last_hidden_state[:, 0, :]
print("CLS embedding shape:", cls_embedding.shape)  # (1, 768)


# 4. GPT: a decoder-only transformer with causal masking.
# Each token can only attend to tokens before it, which makes GPT a natural text generator.
#
#                       BERT (encoder)              GPT (decoder)
#   Attention           Bidirectional               Left-to-right only
#   Best for            Classification, NER, Q&A    Text generation
#   Training objective  Masked Language Modelling   Next-token prediction
#   Examples            BERT, RoBERTa, DeBERTa      GPT-4, Claude, Llama


# 5. Fine-tuning BERT for classification.
# Full IMDB training (25,000 reviews, 3 epochs) needs a GPU and takes a long time.
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load a sentiment dataset
dataset = load_dataset("imdb")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")


# batched=True sends many examples to tokenize() at once, which is much faster.
tokenized = dataset.map(tokenize, batched=True)

# Load BERT with a classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# evaluation_strategy was renamed eval_strategy in newer transformers versions.
training_args = TrainingArguments(
    output_dir="models/bert-sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer runs the training loop: batches, loss, backpropagation and evaluation.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()
