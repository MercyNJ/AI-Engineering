
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report


# .keras is the current Keras save format (.h5 is the older format).
MODEL_PATH = Path("models/mnist_classifier.keras")


def build_model():
    # Sequential stacks layers one after another.
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),                  # 28×28 image input
        keras.layers.Flatten(),                       # flatten 2D image to 1D vector
        keras.layers.Dense(256, activation="relu"),   # hidden layer: 256 neurons
        keras.layers.Dropout(0.3),                    # regularisation: randomly zero 30% of neurons
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"), # output: 10 classes
    ])

    # compile() sets the optimiser, loss function, and metrics before training.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",   # for integer labels
        metrics=["accuracy"],
    )

    # Loss function choices:
    # sparse_categorical_crossentropy -> multi-class, integer labels
    # categorical_crossentropy        -> multi-class, one-hot labels
    # binary_crossentropy             -> binary classification
    # mean_squared_error              -> regression

    return model


# If validation loss increases while training loss decreases,
# the model is overfitting — add Dropout or reduce model size.
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # history.history stores each metric's value per epoch.
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

    (X_train, y_train), (X_test, y_test) = (
        keras.datasets.mnist.load_data()
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Normalise pixel values to [0, 1].
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    model = build_model()
    model.summary()

    # Callbacks — actions taken during training.
    callbacks = [
        # Stop when validation loss stops improving and keep the best weights.
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        # Halve the learning rate when validation loss stops improving.
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.15,    # hold out 15% of training data for validation
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # predict() returns one probability per class for each image.
    probabilities = model.predict(X_test[:5], verbose=0)
    # argmax() picks the class with the highest probability.
    predictions = probabilities.argmax(axis=1)

    print("\nExample predictions:")
    for index, prediction in enumerate(predictions):
        confidence = probabilities[index, prediction]
        print(
            f"Actual: {y_test[index]} | "
            f"Predicted: {prediction} | "
            f"Confidence: {confidence:.1%}"
        )

    all_probabilities = model.predict(X_test, verbose=0)
    all_predictions = all_probabilities.argmax(axis=1)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        all_predictions,
        zero_division=0,
    ))

    # save() stores the architecture, weights, and optimiser state.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Load later and check it gives the same predictions.
    loaded_model = keras.models.load_model(MODEL_PATH)
    loaded_predictions = loaded_model.predict(X_test[:5], verbose=0).argmax(axis=1)

    print(f"Original model predictions: {predictions}")
    print(f"Loaded model predictions:   {loaded_predictions}")

    plot_history(history)


if __name__ == "__main__":
    main()
