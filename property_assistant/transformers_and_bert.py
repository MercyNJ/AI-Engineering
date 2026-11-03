"""
Transformers & BERT

Applies self-attention, BERT contextual embeddings and BERT fine-tuning
to property listings built from properties.csv.

Install: pip install transformers torch datasets accelerate pandas
"""

# 1. Load the properties and turn each row into a text listing.
import os

import numpy as np
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)


def describe_listing(row) -> str:
    furnished = "furnished" if row["furnished"] == "Yes" else "unfurnished"
    return (
        f"{row['bedrooms']} bedroom, {row['bathrooms']} bathroom "
        f"{furnished} {row['property_type']} in {row['property_location']}. "
        f"Price: KES {row['price_kes']:,}."
    )


properties["description"] = properties.apply(describe_listing, axis=1)

print(f"Dataset: {len(properties)} properties")
print("Example listing:", properties["description"].iloc[0])


# 2. BERT contextual embeddings for a listing.
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# attn_implementation="eager" lets BERT return its attention weights in section 3.
bert = BertModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
bert.eval()

listing = properties["description"].iloc[0]
encoding = tokenizer(listing, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = bert(**encoding)

print("\n=== BERT Embeddings for One Listing ===")
print("Tokens:", tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]))
print("Hidden states shape:", outputs.last_hidden_state.shape)
print("CLS embedding shape:", outputs.last_hidden_state[:, 0, :].shape)


# 3. Self-attention inside BERT: what does "it" look at?
request = "I like the Kilimani apartment because it is close to my office."
encoding = tokenizer(request, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

with torch.no_grad():
    outputs = bert(**encoding, output_attentions=True)

# outputs.attentions has one tensor per layer, each shaped (batch, heads, seq_len, seq_len).
last_layer = outputs.attentions[-1][0]
print(f"\nBERT has {len(outputs.attentions)} layers with {last_layer.shape[0]} attention heads each")

# Average the 12 heads to get one attention row per token.
attention = last_layer.mean(dim=0)
it_index = tokens.index("it")

print(f"Request: {request}")
print("Tokens 'it' attends to most (last layer, averaged over heads):")

for index in attention[it_index].argsort(descending=True)[:5]:
    print(f"  {tokens[index]:<10} {attention[it_index, index]:.3f}")


# 4. Attention as a soft search: requests (queries) look at listings (keys).
def self_attention(Q, K, V, scale=True):
    """Scaled dot-product attention."""
    d_k   = Q.shape[-1]
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(d_k)

    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    output = weights @ V
    return output, weights


# Mean pooling: average the token vectors, ignoring padding, to get one vector per text.
def embed_texts(texts):
    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)

    with torch.no_grad():
        outputs = bert(**encoding)

    mask = encoding["attention_mask"].unsqueeze(-1)
    summed = (outputs.last_hidden_state * mask).sum(dim=1)
    return (summed / mask.sum(dim=1)).numpy()


listing_vectors = embed_texts(properties["description"].tolist())

user_requests = [
    "furnished flat in westlands",
    "family home in ruiru",
]
request_vectors = embed_texts(user_requests)

# Here Q comes from the requests, while K and V come from the listings.
# Each row of weights says how much one request attends to each of the 50 listings.
_, weights = self_attention(request_vectors, listing_vectors, listing_vectors)

print("\n=== Top 3 Listings per Request (Attention Weights) ===")
for request, row in zip(user_requests, weights):
    print(f"Request: {request}")

    for index in row.argsort()[::-1][:3]:
        listing_row = properties.iloc[index]
        print(f"  {row[index]:.3f} | #{listing_row['property_id']} {listing_row['description']}")
    print()


# 5. Fine-tune BERT to predict suitability from the listing text.
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Include the user's location so the text carries the same information as the CSV row.
texts = [
    f"User searching in {row.user_location}. {row.description}"
    for row in properties.itertuples()
]
labels = (properties["suitable"] == "Yes").astype(int).tolist()

# Trainer expects the target column to be called "label".
dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.3, seed=42)


def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")


tokenized = dataset.map(tokenize, batched=True)

classifier = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# save_strategy="no" skips checkpoints, because each BERT checkpoint is over 1 GB.
training_args = TrainingArguments(
    output_dir="models/bert-property-suitability",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=classifier,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()

# predict() returns the raw scores (logits); argmax picks the higher-scoring class.
predictions = trainer.predict(tokenized["test"])
predicted_labels = predictions.predictions.argmax(axis=1)
accuracy = (predicted_labels == predictions.label_ids).mean()

print(f"\nValidation accuracy: {accuracy:.1%} on {len(predicted_labels)} properties")
print("With only 50 properties, this accuracy changes a lot between splits.")


# 6. Use the fine-tuned model on a new property text.
new_text = (
    "User searching in Kilimani. 2 bedroom, 2 bathroom furnished Apartment "
    "in Kilimani. Price: KES 80,000."
)

classifier.eval()
encoding = tokenizer(new_text, return_tensors="pt", truncation=True, max_length=64)

# .to() moves the inputs to the same device (CPU or GPU) as the model.
encoding = {key: value.to(classifier.device) for key, value in encoding.items()}

with torch.no_grad():
    logits = classifier(**encoding).logits

# softmax() turns the logits into probabilities that sum to 1.
probabilities = torch.softmax(logits, dim=-1)[0]

print(f"\nNew property: {new_text}")
print(f"Suitable probability: {probabilities[1]:.1%}")
