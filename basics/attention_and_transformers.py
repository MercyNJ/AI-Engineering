
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)


# 1. Scaled dot-product attention

def scaled_dot_product_attention(query, key, value):
    key_dim = tf.cast(tf.shape(key)[-1], tf.float32)
    scores = tf.matmul(query, key, transpose_b=True)
    scores /= tf.sqrt(key_dim)
    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, value)
    return output, weights


query = tf.random.normal((4, 8))
key = tf.random.normal((4, 8))
value = tf.random.normal((4, 8))

output, attention_weights = scaled_dot_product_attention(
    query, key, value
)

print("Attention output shape:", output.shape)
print("Attention weights:\n", attention_weights.numpy().round(2))
print("Row sums:", attention_weights.numpy().sum(axis=-1).round(2))


# 2. Multi-head self-attention

sequence = tf.random.normal((1, 10, 64))

mha = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=16,
)

mha_output, mha_weights = mha(
    sequence,
    sequence,
    return_attention_scores=True,
)

print("\nMulti-head output shape:", mha_output.shape)
print("Attention scores shape:", mha_weights.shape)


# 3. Positional encoding

def get_positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]

    angles = positions / np.power(
        10000,
        (2 * (dimensions // 2)) / d_model,
    )

    encoding = np.zeros_like(angles, dtype=np.float32)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.constant(encoding[np.newaxis, :, :])


positional_encoding = get_positional_encoding(
    seq_len=10,
    d_model=64,
)

sequence_with_positions = sequence + positional_encoding

print("\nPositional encoding shape:", positional_encoding.shape)
print("Sequence with positions:", sequence_with_positions.shape)


# 4. Transformer block

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
        )

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attention_output = self.mha(x, x)
        x = self.norm1(
            x + self.drop1(attention_output, training=training)
        )

        ffn_output = self.ffn(x)
        return self.norm2(
            x + self.drop2(ffn_output, training=training)
        )


transformer = TransformerBlock(
    d_model=64,
    num_heads=4,
    ff_dim=128,
)

transformer_output = transformer(
    sequence_with_positions,
    training=False,
)

print("\nTransformer output shape:", transformer_output.shape)


# 5. Trainable example using generated data

inputs = keras.Input(shape=(10, 64))
x = inputs + get_positional_encoding(10, 64)
x = TransformerBlock(64, 4, 128)(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

x_train = tf.random.normal((300, 10, 64))
y_train = tf.random.uniform(
    (300,),
    minval=0,
    maxval=3,
    dtype=tf.int32,
)

model.fit(x_train, y_train, epochs=3, batch_size=32)

predictions = model.predict(x_train[:5], verbose=0)
print("\nPrediction probabilities:\n", predictions.round(3))

MODEL_PATH = Path("models/attention_transformer_model.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
