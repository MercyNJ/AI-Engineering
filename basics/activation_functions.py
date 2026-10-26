import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# 1. Define the activation functions
# --------------------------------------------------

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    # Clip inputs to avoid overflow in exp for extreme values.
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # Subtracting the maximum improves numerical stability.
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(
        exp_values,
        axis=-1,
        keepdims=True,
    )


def tanh(x):
    return np.tanh(x)


# --------------------------------------------------
# 2. Plot the activation functions
# --------------------------------------------------

x = np.linspace(-8, 8, 400)

plt.figure(figsize=(10, 6))
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Test ReLU
# --------------------------------------------------

print("\nRELU")
relu_inputs = np.array([-4, -2, -1, 0, 1, 2, 4])
print("Inputs: ", relu_inputs)
print("Outputs:", relu(relu_inputs))


# --------------------------------------------------
# 4. Test sigmoid
# --------------------------------------------------

print("\nSIGMOID")
sigmoid_inputs = np.array([-4, -2, 0, 2, 4])
print("Inputs: ", sigmoid_inputs)
print("Outputs:", np.round(sigmoid(sigmoid_inputs), 4))


# --------------------------------------------------
# 5. Test softmax
# --------------------------------------------------

print("\nSOFTMAX")
logits = np.array([2.1, 0.5, 3.8])
probabilities = softmax(logits)

class_names = [
    "Apartment",
    "Townhouse",
    "Standalone house",
]

for name, probability in zip(class_names, probabilities):
    print(f"{name}: {probability:.2%}")

print("Probability sum:", probabilities.sum())
print("Predicted class:", class_names[np.argmax(probabilities)])


# --------------------------------------------------
# 6. Test tanh
# --------------------------------------------------

print("\nTANH")
tanh_inputs = np.array([-4, -2, 0, 2, 4])
print("Inputs: ", tanh_inputs)
print("Outputs:", np.round(tanh(tanh_inputs), 4))


# --------------------------------------------------
# 7. Compare output ranges
# --------------------------------------------------

print("\nOUTPUT RANGES")
print("ReLU:    0 to infinity")
print("Sigmoid: 0 to 1")
print("Softmax: 0 to 1; values sum to 1")
print("Tanh:   -1 to 1")


# --------------------------------------------------
# 8. Demonstrate sigmoid saturation
# --------------------------------------------------

print("\nSIGMOID SATURATION")
large_inputs = np.array([-10, -5, 0, 5, 10])
large_outputs = sigmoid(large_inputs)

for value, output in zip(large_inputs, large_outputs):
    print(f"Input {value:>3}: output {output:.6f}")

print(
    "At extreme inputs, sigmoid changes very little. "
    "Its gradient becomes very small."
)


# --------------------------------------------------
# 9. Demonstrate softmax with a batch of examples
# --------------------------------------------------

print("\nSOFTMAX FOR MULTIPLE EXAMPLES")

batch_logits = np.array([
    [2.1, 0.5, 3.8],
    [3.0, 2.0, 0.1],
    [0.2, 4.0, 1.0],
])

batch_probabilities = softmax(batch_logits)

print("Probabilities:\n", np.round(batch_probabilities, 3))
print("Row sums:", batch_probabilities.sum(axis=1))


# --------------------------------------------------
# 10. Train a real classification model
# --------------------------------------------------

print("\nTRAINING A CLASSIFICATION MODEL")

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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_val = scaler.transform(X_val).astype("float32")

model = keras.Sequential([
    keras.Input(shape=(2,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=0,
)

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0,
)

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.2%}")


# --------------------------------------------------
# 11. Plot training and validation loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. Plot training and validation accuracy
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(
    history.history["val_accuracy"],
    label="Validation accuracy",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 13. Make predictions
# --------------------------------------------------

print("\nPREDICTIONS")

sample_inputs = X_val[:5]
predicted_probabilities = model.predict(
    sample_inputs,
    verbose=0,
).flatten()

predicted_classes = (
    predicted_probabilities >= 0.5
).astype(int)

for index, probability, predicted_class in zip(
    range(len(sample_inputs)),
    predicted_probabilities,
    predicted_classes,
):
    print(
        f"Example {index + 1}: "
        f"probability of class 1 = {probability:.3f}, "
        f"predicted class = {predicted_class}, "
        f"actual class = {int(y_val[index])}"
    )


# --------------------------------------------------
# 14. Keras activation and loss reference
# --------------------------------------------------

print("\nCOMMON ACTIVATION AND LOSS PAIRINGS")

print(
    "Binary classification: "
    "sigmoid + binary_crossentropy"
)

print(
    "Multiclass classification with integer labels: "
    "softmax + sparse_categorical_crossentropy"
)

print(
    "Multiclass classification with one-hot labels: "
    "softmax + categorical_crossentropy"
)

print(
    "Hidden layers: ReLU is a common starting choice."
)

print("\nActivation function demonstrations complete.")
