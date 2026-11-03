"""
Word Embeddings — Teaching Machines What Words Mean

Word embeddings map words to dense vectors where similar words are close together.
Covers Word2Vec, analogies, pretrained GloVe vectors and the Keras Embedding layer.

Install: pip install gensim tensorflow
"""

# 1. Why embeddings? Integer codes carry no meaning, vectors do.
import numpy as np

# Integer encoding: the numbers say nothing about how words relate.
word_ids = {"cat": 1, "dog": 2, "automobile": 3}
print("Integer encoding:", word_ids)

# Embeddings: similar words get similar vectors (illustrative values).
embeddings = {
    "cat": np.array([0.9, 0.1, -0.3]),
    "dog": np.array([0.8, 0.2, -0.2]),
    "automobile": np.array([-0.5, 0.8, 0.9]),
}


# Cosine similarity: 1 means the vectors point the same way, below 0 means opposite.
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print(f"cat vs dog:        {cosine_similarity(embeddings['cat'], embeddings['dog']):.3f}")
print(f"cat vs automobile: {cosine_similarity(embeddings['cat'], embeddings['automobile']):.3f}")


# 2. Word2Vec: learn embeddings by predicting context.
# Skip-gram: given a word, predict its surrounding words.
# CBOW: given surrounding words, predict the centre word (the gensim default).
from gensim.models import Word2Vec

sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "realm"],
    ["the", "prince", "is", "son", "of", "the", "king"],
    ["the", "princess", "is", "daughter", "of", "the", "queen"],
    ["man", "goes", "to", "work", "every", "day"],
    ["woman", "goes", "to", "work", "every", "day"],
]

# vector_size = numbers per word, window = context words on each side,
# min_count=1 keeps every word, even if it appears only once.
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, epochs=100)

# Check similarity
print("\nking-queen similarity:", model.wv.similarity("king", "queen"))
print("king-man   similarity:", model.wv.similarity("king", "man"))

# Find most similar words
print("\nMost similar to 'king':", model.wv.most_similar("king", topn=3))


# 3. The famous king - man + woman = queen.
# positive words are added, negative words are subtracted.
# With only six sentences the result can vary between runs.
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1,
)
print("\nking - man + woman =", result)  # ideally gives 'queen'


# 4. Using pretrained embeddings trained on billions of words.
import gensim.downloader as api

# Download GloVe 100-dimensional embeddings (822MB)
# glove = api.load("glove-wiki-gigaword-100")

# Or use a smaller model for testing
print("\nLoading embeddings...")
word_vectors = api.load("glove-twitter-25")
print("Loaded! Dimensions:", word_vectors.vector_size)

print("\nWords similar to 'python':")
print(word_vectors.most_similar("python", topn=5))
print("\nVector for 'python':", word_vectors["python"][:5], "...")
print("Dimensions:", word_vectors.vector_size)

# Analogy: paris - france + germany = ?
print("\nparis - france + germany =")
print(word_vectors.most_similar(positive=["paris", "germany"], negative=["france"], topn=3))

# Similarity comparisons
print(f"\nhappy vs sad:    {word_vectors.similarity('happy', 'sad'):.3f}")
print(f"happy vs joyful: {word_vectors.similarity('happy', 'joyful'):.3f}")


# 5. Embedding layer in Keras.
from tensorflow import keras

vocab_size  = 10000
embed_dim   = 128
sequence_len = 200

# The Embedding layer is a look-up table: it maps each token index to its vector.
# These vectors are learned during training.
# Keras 3 removed input_length, so keras.Input sets the sequence length instead.
model = keras.Sequential([
    keras.Input(shape=(sequence_len,)),
    keras.layers.Embedding(vocab_size, embed_dim),
    # GlobalAveragePooling1D() averages the word vectors into one vector per text.
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# 6. Loading pretrained GloVe into Keras.
from pathlib import Path


# Assuming word_index is your tokeniser's word_index
def load_glove_matrix(glove_path, word_index, embed_dim=100):
    embeddings = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word   = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector

    # Row i holds the vector for the word with index i. Row 0 is reserved for padding.
    matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        vec = embeddings.get(word)
        if vec is not None:
            matrix[i] = vec
    return matrix


# A small word_index built from the Word2Vec sentences, starting at 1.
vocabulary = sorted({word for sentence in sentences for word in sentence})
word_index = {word: i for i, word in enumerate(vocabulary, start=1)}

# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ and unzip it into data/.
GLOVE_PATH = Path("data/glove.6B.100d.txt")
embed_dim = 100

if GLOVE_PATH.exists():
    embed_matrix = load_glove_matrix(GLOVE_PATH, word_index, embed_dim)
else:
    print(f"\n{GLOVE_PATH} not found. Using an all-zero matrix instead.")
    embed_matrix = np.zeros((len(word_index) + 1, embed_dim))

print("Embedding matrix shape:", embed_matrix.shape)

# Use pretrained weights in Embedding layer
embedding_layer = keras.layers.Embedding(
    len(word_index) + 1, embed_dim,
    embeddings_initializer=keras.initializers.Constant(embed_matrix),
    trainable=False,   # freeze pretrained weights
)
