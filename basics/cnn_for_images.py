
"""
Convolutional Neural Networks (CNNs) for Images
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 1. Create a convolution filter

# A filter is a small grid that slides over an image to detect a pattern.
# This Sobel filter detects vertical edges.
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

print("Convolution filter:")
print(sobel_vertical)


# 2. Load the CIFAR-10 image dataset

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)


# 3. Define the image classes

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Number of classes:", len(class_names))


# 4. Normalize pixel values

# Scale pixels from 0-255 to 0-1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# 5. Build a Convolutional Neural Network

model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # Conv2D learns 32 filters of size 3x3 to detect patterns in the image.
    # padding="same" keeps the output the same width and height as the input.
    layers.Conv2D(32, kernel_size=3, padding="same"),
    layers.Activation("relu"),
    # MaxPooling2D keeps the largest value in each 2x2 area, halving the image size.
    layers.MaxPooling2D(pool_size=2),

    # Later layers use more filters to detect more complex patterns.
    layers.Conv2D(64, kernel_size=3, padding="same"),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(128, kernel_size=3, padding="same"),
    layers.Activation("relu"),

    # Flatten turns the 3D feature maps into a 1D vector for the Dense layers.
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.summary()


# 6. Compile the model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 7. Configure early stopping

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# 8. Train the model

history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping]
)


# 9. Evaluate the model

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# 10. Make a prediction

predictions = model.predict(x_test[:5])

for index, probabilities in enumerate(predictions):
    # argmax() picks the class with the highest probability.
    predicted_class = np.argmax(probabilities)
    actual_class = int(y_test[index][0])

    print(f"\nImage {index + 1}")
    print("Predicted:", class_names[predicted_class])
    print("Actual:", class_names[actual_class])
    print("Confidence:", f"{probabilities[predicted_class]:.2%}")


# 11. Save the trained model

# models/ is in .gitignore, so the saved model is not committed.
MODEL_PATH = Path("models/cnn_image_classifier.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
