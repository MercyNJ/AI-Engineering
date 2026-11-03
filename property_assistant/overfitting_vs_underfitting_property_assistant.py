"""
Overfitting vs Underfitting

Uses generated illustrative data with the project's property feature names.
"""

# 1. Generate illustrative property records using the project dataset's columns.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

n_samples = 700

locations = np.array([
    "Kilimani",
    "Ruiru",
    "Karen",
    "Westlands",
    "Kasarani",
])

property_types = np.array([
    "Apartment",
    "Townhouse",
    "House",
])

user_location = np.random.choice(locations, n_samples)
property_location = np.random.choice(locations, n_samples)
property_type = np.random.choice(property_types, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.clip(
    bedrooms + np.random.randint(-1, 2, n_samples),
    1,
    5,
)
price_kes = np.random.randint(25000, 220001, n_samples)
furnished = np.random.choice(["Yes", "No"], n_samples)


# Create an illustrative suitability label from several property attributes.
location_match = (
    user_location == property_location
).astype(int)

price_score = (
    price_kes <= 100000
).astype(int)

space_score = (
    (bedrooms >= 2) & (bathrooms >= 1)
).astype(int)

furnished_score = (
    furnished == "Yes"
).astype(int)

suitable = (
    location_match
    + price_score
    + space_score
    + furnished_score
    >= 2
).astype(int)


properties = pd.DataFrame({
    "user_location": user_location,
    "property_location": property_location,
    "property_type": property_type,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "price_kes": price_kes,
    "furnished": furnished,
    "suitable": suitable,
})

print("Illustrative property records:")
print(properties.head())

print("\nSuitability class counts:")
print(properties["suitable"].value_counts())


# 2. Split the records into training and validation sets.
X = properties.drop(columns="suitable")
y = properties["suitable"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)


# 3. Encode categorical fields and scale numeric fields.
categorical_features = [
    "user_location",
    "property_location",
    "property_type",
    "furnished",
]

numeric_features = [
    "bedrooms",
    "bathrooms",
    "price_kes",
]

preprocessor = ColumnTransformer([
    (
        "categories",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features,
    ),
    (
        "numbers",
        StandardScaler(),
        numeric_features,
    ),
])

X_train_processed = preprocessor.fit_transform(
    X_train
).astype("float32")

X_val_processed = preprocessor.transform(
    X_val
).astype("float32")

input_size = X_train_processed.shape[1]


# Build a larger property model without regularisation.
def make_overfit_model():
    return keras.Sequential([
        keras.Input(shape=(input_size,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


# Add dropout and L2 regularisation to encourage generalisation.
def make_regularised_model():
    return keras.Sequential([
        keras.Input(shape=(input_size,)),
        keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        ),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        ),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


# Train the larger model without dropout or L2.
overfit_model = make_overfit_model()

overfit_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_overfit = overfit_model.fit(
    X_train_processed,
    y_train,
    validation_data=(X_val_processed, y_val),
    epochs=60,
    batch_size=16,
    verbose=0,
)


# Stop the regularised model when validation loss stops improving.
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

regularised_model = make_regularised_model()

regularised_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_regularised = regularised_model.fit(
    X_train_processed,
    y_train,
    validation_data=(X_val_processed, y_val),
    epochs=60,
    batch_size=16,
    verbose=0,
    callbacks=[early_stop],
)


# Compare training and validation accuracy for both property models.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, history, title in zip(
    axes,
    [history_overfit, history_regularised],
    [
        "Property Model Without Regularisation",
        "Regularised Property Model",
    ],
):
    ax.plot(
        history.history["accuracy"],
        label="Train",
        linewidth=2,
    )

    ax.plot(
        history.history["val_accuracy"],
        label="Validation",
        linewidth=2,
        linestyle="--",
    )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Display the best validation accuracy achieved by each model.
print(
    "Best validation accuracy without regularisation:",
    f"{max(history_overfit.history['val_accuracy']):.1%}",
)

print(
    "Best validation accuracy with regularisation:",
    f"{max(history_regularised.history['val_accuracy']):.1%}",
)

print(
    "\nThese results use generated illustrative data, not real listings."
)