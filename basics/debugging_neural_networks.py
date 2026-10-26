import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# 1. Generate a small classification dataset
# --------------------------------------------------

X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=SEED,
)

X = X.astype("float32")
y = y.astype("float32")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

print("\nDataset shapes")
print("Training inputs:", X_train.shape)
print("Training labels:", y_train.shape)
print("Validation inputs:", X_val.shape)
print("Validation labels:", y_val.shape)


# --------------------------------------------------
# 2. Inspect the data pipeline
# --------------------------------------------------

def inspect_data(inputs, labels, count=5):
    print("\nSample input-label pairs")

    for index in range(min(count, len(inputs))):
        print(
            f"Input: {inputs[index]}, "
            f"Label: {labels[index]}"
        )

    print("\nInput values finite:", np.isfinite(inputs).all())
    print("Labels finite:", np.isfinite(labels).all())
    print("Unique labels:", np.unique(labels))


inspect_data(X_train, y_train)


# --------------------------------------------------
# 3. Build a suitable binary classification model
# --------------------------------------------------

def build_binary_classifier(
    input_dim=2,
    learning_rate=0.001,
    dropout_rate=0.0,
    weight_decay=0.0,
):
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=(
                    keras.regularizers.L2(weight_decay)
                    if weight_decay > 0
                    else None
                ),
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------
# 4. Train and plot both loss curves
# --------------------------------------------------

def plot_loss(history, title):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training loss",
    )

    if "val_loss" in history.history:
        plt.plot(
            history.history["val_loss"],
            label="Validation loss",
        )

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.show()


print("\nTraining a baseline model...")

model = build_binary_classifier()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=0,
)

plot_loss(history, "Baseline: Training vs Validation Loss")

train_loss, train_accuracy = model.evaluate(
    X_train,
    y_train,
    verbose=0,
)

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0,
)

print(f"Training loss: {train_loss:.4f}")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")


# --------------------------------------------------
# 5. Test whether the model can overfit 10 examples
# --------------------------------------------------

print("\nTesting whether the model can memorise 10 examples...")

X_tiny = X_train[:10]
y_tiny = y_train[:10]

tiny_model = build_binary_classifier(
    learning_rate=0.001,
)

tiny_history = tiny_model.fit(
    X_tiny,
    y_tiny,
    epochs=300,
    batch_size=10,
    verbose=0,
)

tiny_predictions = tiny_model.predict(
    X_tiny,
    verbose=0,
).reshape(-1)

tiny_classes = (tiny_predictions >= 0.5).astype("float32")
tiny_accuracy = np.mean(tiny_classes == y_tiny)

print(f"Accuracy on the same 10 examples: {tiny_accuracy:.2%}")
print("Actual labels:   ", y_tiny.astype(int))
print("Predicted labels:", tiny_classes.astype(int))

plot_loss(
    tiny_history,
    "Tiny Dataset: Can the Model Memorise 10 Examples?",
)


# --------------------------------------------------
# 6. Demonstrate input normalisation
# --------------------------------------------------

print("\nDemonstrating input normalisation...")

X_unscaled = np.column_stack(
    [
        np.random.uniform(0, 100, 1000),
        np.random.uniform(0, 10_000_000, 1000),
    ]
).astype("float32")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_unscaled)

print("Unscaled feature means:", X_unscaled.mean(axis=0))
print("Unscaled feature stds: ", X_unscaled.std(axis=0))
print("Scaled feature means:  ", X_scaled.mean(axis=0))
print("Scaled feature stds:   ", X_scaled.std(axis=0))

# Fit the scaler on training data only in a real workflow.
# Reuse that fitted scaler to transform validation and test data.


# --------------------------------------------------
# 7. Demonstrate learning-rate choices
# --------------------------------------------------

def run_learning_rate_experiment(learning_rate):
    experiment_model = build_binary_classifier(
        learning_rate=learning_rate,
    )

    experiment_history = experiment_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        verbose=0,
    )

    return experiment_history


print("\nComparing learning rates...")

learning_rates = [0.0001, 0.001, 0.1]
learning_rate_histories = {}

for learning_rate in learning_rates:
    experiment_history = run_learning_rate_experiment(
        learning_rate
    )

    learning_rate_histories[learning_rate] = experiment_history

    final_loss = experiment_history.history["loss"][-1]

    print(
        f"Learning rate {learning_rate}: "
        f"final training loss = {final_loss:.4f}"
    )

plt.figure(figsize=(9, 5))

for learning_rate, experiment_history in learning_rate_histories.items():
    plt.plot(
        experiment_history.history["loss"],
        label=f"Learning rate {learning_rate}",
    )

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Effect of Learning Rate on Training")
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. Demonstrate detecting NaN and infinite values
# --------------------------------------------------

print("\nChecking for numerical instability...")

normal_values = np.array([1.0, 2.0, 3.0])
bad_values = np.array([1.0, np.nan, np.inf])

print("Normal values finite:", np.isfinite(normal_values).all())
print("Bad values finite:", np.isfinite(bad_values).all())


def check_loss_values(loss_values):
    loss_values = np.asarray(loss_values)

    if not np.isfinite(loss_values).all():
        print(
            "WARNING: Loss contains NaN or infinity. "
            "Inspect inputs, labels, the loss function, "
            "and the learning rate."
        )
    else:
        print("Loss values are finite.")


check_loss_values([0.9, 0.7, 0.5])
check_loss_values([0.9, np.nan, 0.5])


# --------------------------------------------------
# 9. Demonstrate a correct loss/output pairing
# --------------------------------------------------

print("\nCorrect binary classification pairing:")
print("Output layer: Dense(1, activation='sigmoid')")
print("Loss: binary_crossentropy")

print("\nCorrect multiclass classification pairing:")
print("Output layer: Dense(number_of_classes, activation='softmax')")
print("Loss: sparse_categorical_crossentropy for integer labels")

print("\nImportant:")
print(
    "A model can sometimes run with an unsuitable loss/output "
    "combination without crashing. Check that the loss matches "
    "the output layer and the format of the labels."
)


# --------------------------------------------------
# 10. Regularisation options for overfitting
# --------------------------------------------------

print("\nBuilding a regularised model...")

regularised_model = build_binary_classifier(
    learning_rate=0.001,
    dropout_rate=0.3,
    weight_decay=0.0001,
)

regularised_history = regularised_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=0,
)

plot_loss(
    regularised_history,
    "Regularised Model: Training vs Validation Loss",
)


# --------------------------------------------------
# 11. Automated loss-curve warning
# --------------------------------------------------

def diagnose_loss(history):
    train_losses = np.asarray(history.history["loss"])
    val_losses = np.asarray(history.history.get("val_loss", []))

    print("\nBasic loss-curve checks")

    if not np.isfinite(train_losses).all():
        print("- Training loss contains NaN or infinity.")
        print("- Check learning rate, inputs and custom loss calculations.")
        return

    if len(train_losses) >= 5:
        recent = train_losses[-5:]
        earlier = train_losses[:-5]

        if len(earlier) and recent.mean() >= earlier[-1]:
            print("- Training loss may have stalled or increased.")
            print("- Inspect the curve before changing the model.")

    if len(val_losses) >= 5:
        if (
            train_losses[-1] < train_losses[0]
            and val_losses[-1] > val_losses[0]
        ):
            print("- Validation loss increased while training progressed.")
            print("- Possible overfitting; inspect both curves.")

    print(
        "These are clues, not definitive diagnoses. "
        "Use the full curves and data checks."
    )


diagnose_loss(history)

print("\nDebugging demonstrations complete.")