"""
Overfitting vs Underfitting — Finding the Sweet Spot

Underfitting: The model is too simple to learn the important patterns
in the training data.

Overfitting: The model learns the training data too closely, including
noise, and performs poorly on unseen data.

General machine-learning examples.
"""

# 1. Visualise underfitting, a good fit, and overfitting with polynomial regression.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(42)
X = np.linspace(0, 10, 30)
y = 2 * X + np.sin(X) + np.random.normal(0, 1.5, 30)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, degree, title in zip(
    axes,
    [1, 3, 15],
    ["Underfitting (degree 1)", "Good fit (degree 3)", "Overfitting (degree 15)"],
):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("lr", LinearRegression()),
    ])
    model.fit(X.reshape(-1, 1), y)

    x_plot = np.linspace(0, 10, 200)
    ax.scatter(X, y, alpha=0.6, color="#f97316", s=40, label="Data")
    ax.plot(
        x_plot,
        model.predict(x_plot.reshape(-1, 1)),
        color="#6366f1",
        linewidth=2,
        label="Model",
    )
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 2. Create a small classification dataset for a neural-network comparison.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

np.random.seed(42)
tf.random.set_seed(42)

X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=8,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Build a large network without regularisation to illustrate overfitting.
def make_overfit_model():
    return keras.Sequential([
        keras.Input(shape=(20,)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


# Use dropout and L2 penalties to encourage better generalisation.
def make_regularised_model():
    return keras.Sequential([
        keras.Input(shape=(20,)),
        keras.layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        ),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(
            512,
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
        keras.layers.Dense(1, activation="sigmoid"),
    ])


# Train the model without regularisation.
m1 = make_overfit_model()
m1.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

h1 = m1.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    verbose=0,
)


# Stop regularised-model training when validation loss stops improving.
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

m2 = make_regularised_model()
m2.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

h2 = m2.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    verbose=0,
    callbacks=[early_stop],
)


# Compare training and validation accuracy across epochs.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, history, title in zip(
    axes,
    [h1, h2],
    ["Overfit Model", "Regularised Model"],
):
    ax.plot(history.history["accuracy"], label="Train", linewidth=2)
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

best_overfit_val = max(h1.history["val_accuracy"])
best_regularised_val = max(h2.history["val_accuracy"])

print(f"Overfit model best validation accuracy:     {best_overfit_val:.1%}")
print(f"Regularised model best validation accuracy: {best_regularised_val:.1%}")
print(
    f"Difference: "
    f"{(best_regularised_val - best_overfit_val) * 100:+.1f} percentage points"
)


# 3. Standalone Dropout example for a multiclass neural network.
dropout_example = keras.Sequential([
    keras.Input(shape=(20,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax"),
])


# 4. Standalone L2 regularisation example for a multiclass neural network.
l2_example = keras.Sequential([
    keras.Input(shape=(20,)),
    keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
    ),
    keras.layers.Dense(10, activation="softmax"),
])


# 5. Early stopping example for an already compiled model.
early_stop_example = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

print("Examples defined. The comparison models were trained above.")