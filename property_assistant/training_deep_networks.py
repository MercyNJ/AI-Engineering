
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path("models/property_suitability_model.keras")
SCALER_PATH = Path("models/property_feature_scaler.joblib")
COLUMNS_PATH = Path("models/property_feature_columns.joblib")

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")


# 1. Build an illustrative property dataset

# properties.csv has only 50 rows, which is too few for a deep network,
# so generate a larger dataset using the same locations and price_kes scale.
def build_property_dataset(n=2000, seed=42):
    properties = pd.read_csv(CSV_PATH)
    locations = sorted(properties["property_location"].unique())

    rng = np.random.default_rng(seed)

    bedrooms = rng.choice([2, 3, 4], size=n, p=[0.5, 0.4, 0.1])
    bathrooms = np.where(bedrooms == 2, rng.integers(1, 3, n), bedrooms - 1)

    price_kes = np.clip(
        rng.normal(30_000 * bedrooms + 10_000, 20_000),
        40_000,
        220_000,
    ).round(-3)

    budget_kes = rng.uniform(40_000, 200_000, n).round(-3)

    # Suitable when the price is within the user's budget.
    suitable = (price_kes <= budget_kes).astype(int)

    # Flip 8% of labels so the data is not perfectly clean.
    noisy_indices = rng.choice(n, size=int(n * 0.08), replace=False)
    suitable[noisy_indices] = 1 - suitable[noisy_indices]

    return pd.DataFrame({
        "user_location": rng.choice(locations, n),
        "property_location": rng.choice(locations, n),
        "property_type": np.where(bedrooms == 4, "Maisonette", "Apartment"),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "price_kes": price_kes,
        "budget_kes": budget_kes,
        "furnished": rng.choice(["Yes", "No"], n),
        "suitable": suitable,
    })


df = build_property_dataset()

feature_columns = [
    "user_location",
    "property_location",
    "property_type",
    "bedrooms",
    "bathrooms",
    "price_kes",
    "budget_kes",
    "furnished",
]

target_column = "suitable"

df = df[feature_columns + [target_column]].dropna().copy()

print(f"Dataset: {len(df)} properties")
print(f"Suitable: {df[target_column].sum()}")


# 2. Prepare categorical and numerical features

# get_dummies() turns each category into its own 0/1 column.
X = pd.get_dummies(
    df[feature_columns],
    columns=[
        "user_location",
        "property_location",
        "property_type",
        "furnished",
    ],
    dtype=float
)

y = df[target_column].astype("float32")

if set(y.unique()) != {0.0, 1.0}:
    raise ValueError(
        "The 'suitable' column must contain binary labels 0 and 1."
    )


# 3. Split the data before fitting the scaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit the scaler on training data only, so the test data stays unseen.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")


# 4. Build a regularised suitability classifier

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),

    # he_normal sets starting weights that suit ReLU layers.
    layers.Dense(
        128,
        kernel_initializer="he_normal"
    ),
    # BatchNormalization rescales each layer's outputs, which keeps training stable.
    # It goes before the activation, so Dense has no activation of its own.
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    layers.Dense(
        64,
        kernel_initializer="he_normal"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.2),

    # glorot_uniform suits sigmoid and softmax output layers.
    layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="glorot_uniform"
    )
])

model.summary()


# 5. Compile the model with gradient clipping

# clipnorm=1.0 limits the size of each weight update, so one bad batch can't destabilise training.
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# 6. Configure training callbacks

callbacks = [
    # Halve the learning rate when validation loss stops improving, but never go below min_lr.
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )
]


# 7. Train the model

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)


# 8. Evaluate on the held-out test set

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# 9. Predict suitability

probabilities = model.predict(X_test[:5]).flatten()

for index, probability in enumerate(probabilities):
    prediction = "Suitable" if probability >= 0.5 else "Not suitable"

    print(
        f"Property {index + 1}: {prediction} "
        f"(probability={probability:.2%})"
    )


# 10. Save the trained model and preprocessing objects

# New properties must use the same scaler and the same dummy columns.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(list(X.columns), COLUMNS_PATH)

print(f"\nModel saved to {MODEL_PATH}")

print(
    "\nThese results use synthetic data. "
    "They are for learning, not real property recommendations."
)
