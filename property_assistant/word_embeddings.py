"""
Word Embeddings

Applies Word2Vec, pretrained GloVe vectors and the Keras Embedding layer
to property listings built from properties.csv.

Install: pip install gensim tensorflow scikit-learn pandas
"""

# 1. Load the properties and turn each row into a text listing.
import os
import re

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


# re.findall() keeps only the words, so numbers and punctuation are dropped.
def to_words(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())


print(f"Dataset: {len(properties)} properties")
print("Example listing:", properties["description"].iloc[0])
print("As words:       ", to_words(properties["description"].iloc[0]))


# 2. Word2Vec: learn embeddings from the property listings themselves.
from gensim.models import Word2Vec

listing_sentences = [to_words(text) for text in properties["description"]]

# workers=1 with a seed makes the result repeatable.
w2v_model = Word2Vec(
    listing_sentences,
    vector_size=25,
    window=3,
    min_count=1,
    epochs=200,
    seed=42,
    workers=1,
)

print("\n=== Word2Vec Trained on Listings ===")
print("Most similar to 'apartment':", w2v_model.wv.most_similar("apartment", topn=3))
print("Most similar to 'kilimani': ", w2v_model.wv.most_similar("kilimani", topn=3))

# Embeddings learn from context: words used in the same position look similar,
# even when their meanings are opposite.
print(f"furnished vs unfurnished: {w2v_model.wv.similarity('furnished', 'unfurnished'):.3f}")


# 3. Pretrained embeddings: meaning learned from billions of words.
import gensim.downloader as api

print("\nLoading embeddings...")
word_vectors = api.load("glove-twitter-25")
print("Loaded! Dimensions:", word_vectors.vector_size)

print("\n=== Pretrained Similarity for Property Words ===")
word_pairs = [
    ("apartment", "flat"),
    ("house", "home"),
    ("apartment", "house"),
    ("furnished", "unfurnished"),
]

for word_a, word_b in word_pairs:
    # similarity() raises an error if a word is not in the vocabulary.
    if word_a in word_vectors and word_b in word_vectors:
        print(f"  {word_a} vs {word_b}: {word_vectors.similarity(word_a, word_b):.3f}")
    else:
        print(f"  {word_a} vs {word_b}: not in vocabulary")

# Unlike subword tokenisers, a word embedding has no vector for words outside its vocabulary.
locations = sorted(properties["property_location"].unique())
missing = [
    location for location in locations
    if any(word not in word_vectors for word in to_words(location))
]
print("\nLocations with no pretrained vector:", missing if missing else "none")


# 4. Match user requests to listings by meaning, not exact keywords.
from sklearn.metrics.pairwise import cosine_similarity


# Average the word vectors to get one vector for the whole text.
def text_vector(text: str) -> np.ndarray:
    words = [word for word in to_words(text) if word in word_vectors]
    if not words:
        return np.zeros(word_vectors.vector_size)
    return np.mean([word_vectors[word] for word in words], axis=0)


listing_vectors = np.array([text_vector(text) for text in properties["description"]])

# "flat" and "home" never appear in the listings, so TF-IDF could not match them.
user_requests = [
    "furnished flat in westlands",
    "family home in ruiru",
]

request_vectors = np.array([text_vector(request) for request in user_requests])
similarities = cosine_similarity(request_vectors, listing_vectors)

print("\n=== Top 3 Listings per Request (Averaged Embeddings) ===")
for request, scores in zip(user_requests, similarities):
    print(f"Request: {request}")

    top_indices = scores.argsort()[::-1][:3]

    for index in top_indices:
        row = properties.iloc[index]
        print(f"  {scores[index]:.2f} | #{row['property_id']} {row['description']}")
    print()


# 5. Keras Embedding layer: predict suitability from the listing text.
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

keras.utils.set_random_seed(42)

SEQUENCE_LEN = 20

# Include the user's location so the text carries the same information as the CSV row.
texts = [
    f"User searching in {row.user_location}. {row.description}"
    for row in properties.itertuples()
]
y = (properties["suitable"] == "Yes").astype("int32").values

# TextVectorization builds the word_index and turns each text into a padded list of token IDs.
vectorizer = keras.layers.TextVectorization(output_sequence_length=SEQUENCE_LEN)
vectorizer.adapt(texts)

X = vectorizer(tf.constant(texts)).numpy()
vocabulary = vectorizer.get_vocabulary()

print("=== Text as Token IDs ===")
print("Vocabulary size:", len(vocabulary))
print("First text:", texts[0])
print("Token IDs: ", X[0])

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)


def build_model(embedding_layer):
    return keras.Sequential([
        keras.Input(shape=(SEQUENCE_LEN,)),
        embedding_layer,
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


# 6. Load the pretrained vectors into the Embedding layer.
# Index 0 is padding and index 1 is [UNK], so they stay as zero vectors.
embed_matrix = np.zeros((len(vocabulary), word_vectors.vector_size))
found = 0

for i, word in enumerate(vocabulary):
    if word in word_vectors:
        embed_matrix[i] = word_vectors[word]
        found += 1

print(f"\nPretrained vectors found for {found} of {len(vocabulary)} words")

models = {
    "Learned embeddings": build_model(
        keras.layers.Embedding(len(vocabulary), 16)
    ),
    "Frozen pretrained embeddings": build_model(
        keras.layers.Embedding(
            len(vocabulary),
            word_vectors.vector_size,
            embeddings_initializer=keras.initializers.Constant(embed_matrix),
            trainable=False,   # freeze pretrained weights
        )
    ),
}

print("\n=== Suitability Model Validation Accuracy ===")
for name, model in models.items():
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        verbose=0,
    )
    _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"  {name}: {val_accuracy:.1%}")

print("\nWith only 50 properties (15 for validation), these accuracies change a lot between splits.")
