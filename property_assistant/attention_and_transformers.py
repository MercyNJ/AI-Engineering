
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

NUM_PROPERTIES = 1200

locations = np.random.randint(0, 5, NUM_PROPERTIES)
property_types = np.random.randint(0, 4, NUM_PROPERTIES)
bedrooms = np.random.randint(1, 6, NUM_PROPERTIES)
bathrooms = np.random.randint(1, 4, NUM_PROPERTIES)
# Same price_kes scale as properties.csv
prices = np.random.randint(40000, 220001, NUM_PROPERTIES)
budgets = np.random.randint(40000, 200001, NUM_PROPERTIES)
furnished = np.random.randint(0, 2, NUM_PROPERTIES)

# Generate illustrative labels for a learning exercise.
suitable = (
    (bedrooms >= 2)
    & (prices <= budgets)
    & (locations <= 3)
).astype(np.int32)

# Each row has seven property features.
features = np.column_stack([
    locations,
    property_types,
    bedrooms,
    bathrooms,
    prices,
    budgets,
    furnished,
]).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    suitable,
    test_size=0.2,
    random_state=42,
    stratify=suitable,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

NUM_FEATURES = X_train.shape[1]
D_MODEL = 32


class FeatureTokenizer(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.projection = layers.Dense(d_model)

    def call(self, inputs):
        return self.projection(inputs[..., tf.newaxis])


class PropertyAttentionBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
        )

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        attention_output = self.attention(x, x)
        x = self.norm1(
            x + self.dropout(attention_output, training=training)
        )

        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)


inputs = keras.Input(shape=(NUM_FEATURES,))

x = FeatureTokenizer(D_MODEL)(inputs)
x = PropertyAttentionBlock(
    d_model=D_MODEL,
    num_heads=4,
    ff_dim=64,
)(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

model.summary()

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

results = model.evaluate(X_test, y_test, verbose=0)
print("\nTest metrics:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

probabilities = model.predict(X_test[:5], verbose=0).flatten()

print("\nSample predictions:")
for probability in probabilities:
    print(
        f"Suitable probability: {probability:.3f} | "
        f"Predicted label: {int(probability >= 0.5)}"
    )

MODEL_PATH = Path("models/property_attention_model.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
