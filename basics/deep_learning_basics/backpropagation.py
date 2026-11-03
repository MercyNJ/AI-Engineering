
"""
Backpropagation — How Neural Networks Learn

Covers:
- Forward pass, loss calculation, and backward pass
- Gradients and gradient descent
- Learning-rate comparisons
- A simple loss-function visualisation
- Training a neural network with TensorFlow/Keras
- Epochs, batches, and training history
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


RANDOM_STATE = 42


# ---------------------------------------------------------
# 1. A simple gradient descent example
# ---------------------------------------------------------

def loss(w):
    """A loss function with its minimum at w = 3."""
    return (w - 3) ** 2


def gradient(w):
    """Derivative of the loss function with respect to w."""
    return 2 * (w - 3)


def demonstrate_gradient_descent():
    print("\n=== 1. GRADIENT DESCENT ===")

    w = 0.0
    learning_rate = 0.1
    history = []

    for step in range(20):
        current_loss = loss(w)
        grad = gradient(w)

        history.append((step, w, current_loss))

        # Move in the direction that reduces the loss.
        w = w - learning_rate * grad

        if step % 5 == 0:
            print(
                f"Step {step:2d}: "
                f"w={history[-1][1]:.4f}, "
                f"loss={current_loss:.4f}, "
                f"gradient={grad:.4f}"
            )

    print(f"\nFinal weight: {w:.4f}")
    print("Target weight: 3.0000")

    return history


# ---------------------------------------------------------
# 2. Compare learning rates
# ---------------------------------------------------------

def run_gradient_descent(learning_rate, steps=30):
    w = 0.0
    weights = []
    losses = []

    for _ in range(steps):
        weights.append(w)
        losses.append(loss(w))

        w -= learning_rate * gradient(w)

    return np.array(weights), np.array(losses)


def compare_learning_rates():
    print("\n=== 2. LEARNING RATE EXPERIMENT ===")

    learning_rates = [0.01, 0.1, 0.9, 1.05]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lr in learning_rates:
        weights, losses = run_gradient_descent(lr)

        axes[0].plot(
            losses,
            label=f"Learning rate = {lr}",
        )

        axes[1].plot(
            weights,
            label=f"Learning rate = {lr}",
        )

        print(
            f"Learning rate {lr:>4}: "
            f"initial loss={losses[0]:.4f}, "
            f"final recorded loss={losses[-1]:.4f}, "
            f"final weight={weights[-1]:.4f}"
        )

    axes[0].set_title("Loss During Training")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(
        y=3,
        linestyle="--",
        label="Target weight = 3",
    )
    axes[1].set_title("Weight During Training")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Weight value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(
        "\nInterpretation:\n"
        "- A small learning rate moves slowly.\n"
        "- A moderate learning rate approaches the minimum smoothly.\n"
        "- A large learning rate can overshoot or oscillate.\n"
        "- An excessively large learning rate can diverge."
    )


# ---------------------------------------------------------
# 3. Visualise the loss function and gradient descent
# ---------------------------------------------------------

def visualise_loss_function():
    print("\n=== 3. LOSS FUNCTION VISUALISATION ===")

    weight_values = np.linspace(-1, 7, 300)
    loss_values = loss(weight_values)

    weights, losses = run_gradient_descent(
        learning_rate=0.1,
        steps=15,
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        weight_values,
        loss_values,
        label="Loss function",
    )

    plt.plot(
        weights,
        losses,
        "o-",
        label="Gradient descent steps",
    )

    plt.scatter(
        [3],
        [0],
        marker="*",
        s=180,
        label="Minimum loss",
    )

    plt.xlabel("Weight value")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Moving Toward Minimum Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4. Create a dataset for a neural network
# ---------------------------------------------------------

def create_dataset():
    """
    Generate a binary classification dataset.

    Each example has two numerical features and belongs to
    one of two classes.
    """
    X, y = make_moons(
        n_samples=1200,
        noise=0.25,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()

    # Fit preprocessing only on training data.
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 5. Build and train the neural network
# ---------------------------------------------------------

def train_neural_network():
    print("\n=== 4. TRAINING A NEURAL NETWORK ===")

    X_train, X_test, y_train, y_test = create_dataset()

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    print("\nTraining begins...")
    print(
        "Keras calculates predictions, measures loss, "
        "uses backpropagation to calculate gradients, "
        "and updates the weights through Adam."
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print("\n=== TEST RESULTS ===")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    plot_training_history(history)

    return model, X_test, y_test, history


# ---------------------------------------------------------
# 6. Plot loss and accuracy during training
# ---------------------------------------------------------

def plot_training_history(history):
    print("\n=== 5. TRAINING HISTORY ===")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        history.history["loss"],
        label="Training loss",
    )
    axes[0].plot(
        history.history["val_loss"],
        label="Validation loss",
    )
    axes[0].set_title("Loss During Neural Network Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history.history["accuracy"],
        label="Training accuracy",
    )
    axes[1].plot(
        history.history["val_accuracy"],
        label="Validation accuracy",
    )
    axes[1].set_title("Accuracy During Neural Network Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(
        "\nLook for decreasing loss. If training loss decreases "
        "while validation loss rises, the model may be overfitting."
    )


# ---------------------------------------------------------
# 7. Make predictions with the trained model
# ---------------------------------------------------------

def demonstrate_predictions(model, X_test, y_test):
    print("\n=== 6. PREDICTIONS AFTER TRAINING ===")

    probabilities = model.predict(X_test[:10], verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    for index, (probability, prediction) in enumerate(
        zip(probabilities, predictions)
    ):
        actual = int(y_test[index])

        print(
            f"Example {index + 1:2d}: "
            f"probability={probability:.3f}, "
            f"predicted class={prediction}, "
            f"actual class={actual}"
        )


# ---------------------------------------------------------
# 8. Key concepts
# ---------------------------------------------------------

def print_key_terms():
    print("\n=== 7. KEY TERMS ===")

    terms = {
        "Forward pass": (
            "Sending input through the network to produce a prediction."
        ),
        "Loss": (
            "A numerical measure of how wrong the prediction is."
        ),
        "Gradient": (
            "The derivative showing how loss changes when a parameter changes."
        ),
        "Backpropagation": (
            "Efficiently calculating gradients for the network's parameters."
        ),
        "Gradient descent": (
            "Using gradients to adjust parameters to reduce loss."
        ),
        "Learning rate": (
            "The size of the parameter update."
        ),
        "Epoch": (
            "One complete pass through the training dataset."
        ),
        "Batch": (
            "A subset of training examples used to calculate an update."
        ),
        "Optimizer": (
            "The algorithm that uses gradients to update parameters; Adam is one example."
        ),
        "Hyperparameter": (
            "A training setting chosen before or during training, such as learning rate."
        ),
    }

    for term, definition in terms.items():
        print(f"{term}: {definition}")


def main():
    demonstrate_gradient_descent()
    compare_learning_rates()
    visualise_loss_function()

    model, X_test, y_test, _ = train_neural_network()
    demonstrate_predictions(model, X_test, y_test)
    print_key_terms()

    print("\nLesson complete.")
    print(
        "Remember: backpropagation calculates gradients; "
        "the optimizer uses those gradients to update the weights."
    )


if __name__ == "__main__":
    main()
