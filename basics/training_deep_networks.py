
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# 1. Load the MNIST dataset

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# reshape(-1, 784) flattens each 28x28 image into 784 values.
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0


# 2. Build a model with Batch Normalisation and Dropout

model = keras.Sequential([
    layers.Input(shape=(784,)),

    # he_normal sets starting weights that suit ReLU layers.
    layers.Dense(
        256,
        kernel_initializer="he_normal"
    ),
    # BatchNormalization rescales each layer's outputs, which keeps training stable.
    # It goes before the activation, so Dense has no activation of its own.
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    layers.Dense(
        128,
        kernel_initializer="he_normal"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    # glorot_uniform suits sigmoid and softmax output layers.
    layers.Dense(
        10,
        activation="softmax",
        kernel_initializer="glorot_uniform"
    )
])

model.summary()


# 3. Configure the optimiser with gradient clipping

# clipnorm=1.0 limits the size of each weight update, so one bad batch can't destabilise training.
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 4. Configure learning-rate reduction and early stopping

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


# 5. Train the model

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks
)


# 6. Evaluate the model

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# 7. Make predictions

predictions = model.predict(X_test[:5])

for index, prediction in enumerate(predictions):
    predicted_digit = prediction.argmax()
    actual_digit = y_test[index]

    print(
        f"Image {index + 1}: "
        f"predicted={predicted_digit}, "
        f"actual={actual_digit}"
    )


# 8. Save the model

MODEL_PATH = Path("models/trained_mnist_model.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
