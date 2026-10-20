
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path("models/property_match_classifier.keras")
SCALER_PATH = Path("models/property_scaler.joblib")

FEATURE_COLUMNS = [
    "price_kes",
    "budget_kes",
    "bedrooms",
    "area_sqm",
    "distance_km",
]


# properties.csv has only 50 rows, which is too few for a neural network,
# so generate a larger illustrative dataset using the same price_kes scale.
def build_property_dataset(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    price_kes = rng.uniform(30_000, 220_000, n).round(-3)
    budget_kes = rng.uniform(40_000, 200_000, n).round(-3)
    bedrooms = rng.integers(1, 6, n)

    area_sqm = np.maximum(
        25,
        bedrooms * rng.normal(32, 6, n)
        + rng.normal(10, 14, n),
    ).round(1)

    distance_km = rng.uniform(1, 30, n).round(1)

    suitable = (
        (price_kes <= budget_kes)
        & (bedrooms >= 2)
        & (area_sqm >= 60)
        & (distance_km <= 20)
    ).astype(int)

    # Flip 8% of labels so the data is not perfectly clean.
    noisy_indices = rng.choice(
        n,
        size=int(n * 0.08),
        replace=False,
    )
    suitable[noisy_indices] = 1 - suitable[noisy_indices]

    return pd.DataFrame({
        "price_kes": price_kes,
        "budget_kes": budget_kes,
        "bedrooms": bedrooms,
        "area_sqm": area_sqm,
        "distance_km": distance_km,
        "suitable_match": suitable,
    })


def build_model(number_of_features):
    model = keras.Sequential([
        keras.Input(shape=(number_of_features,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),                   # randomly zero 20% of neurons
        keras.layers.Dense(16, activation="relu"),
        # sigmoid outputs one probability between 0 and 1 for "suitable".
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",   # for two classes: suitable / not suitable
        metrics=["accuracy"],
    )

    return model


# If validation loss increases while training loss decreases,
# the model is overfitting.
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Training")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Training")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    tf.keras.utils.set_random_seed(42)

    df = build_property_dataset()

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["suitable_match"].to_numpy(dtype=np.float32)

    print(f"Dataset: {X.shape}")
    print(f"Suitable matches: {int(y.sum())}")
    print(f"Unsuitable matches: {int(len(y) - y.sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Neural networks train better when features are on a similar scale.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    model = build_model(len(FEATURE_COLUMNS))
    model.summary()

    callbacks = [
        # restore_best_weights keeps the epoch with the lowest validation loss.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    # With one sigmoid output, use a 0.5 threshold instead of argmax().
    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not suitable", "Suitable"],
        zero_division=0,
    ))

    print("\nExample property predictions:")
    for index in range(min(5, len(probabilities))):
        print(
            f"Property {index + 1}: "
            f"{probabilities[index]:.1%} estimated match probability"
        )

    # Save the scaler too: new properties must be scaled the same way.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

    # Load the saved model and scaler, then score a new property.
    loaded_model = keras.models.load_model(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)

    new_property = pd.DataFrame([{
        "price_kes": 75_000,
        "budget_kes": 90_000,
        "bedrooms": 2,
        "area_sqm": 70,
        "distance_km": 8,
    }])

    new_property_scaled = loaded_scaler.transform(
        new_property[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    )
    match_probability = loaded_model.predict(
        new_property_scaled,
        verbose=0,
    )[0, 0]

    print("\nNew property (loaded model):")
    print(new_property.to_string(index=False))
    print(f"Estimated match probability: {match_probability:.1%}")

    plot_history(history)

    print(
        "\nThese results use synthetic data. "
        "They are for learning, not real property recommendations."
    )


if __name__ == "__main__":
    main()
