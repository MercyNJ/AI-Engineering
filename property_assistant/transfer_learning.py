
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

MODEL_PATH = Path("models/property_image_classifier.keras")
CLASSES_PATH = Path("models/property_image_classes.txt")

class_names = ["bathroom", "bedroom", "kitchen"]
num_classes = len(class_names)


# each room type gets its own main colour plus random noise.
def build_sample_images(images_per_class=200, seed=SEED):
    rng = np.random.default_rng(seed)

    room_colours = [
        [90, 140, 200],    # bathroom: blue
        [180, 120, 80],    # bedroom: warm brown
        [220, 220, 210],   # kitchen: white
    ]

    images = []
    labels = []

    for label, colour in enumerate(room_colours):
        noise = rng.normal(0, 40, (images_per_class, 32, 32, 3))
        images.append(np.clip(np.array(colour) + noise, 0, 255))
        labels.append(np.full(images_per_class, label))

    images = np.concatenate(images).astype("float32")
    labels = np.concatenate(labels)

    order = rng.permutation(len(labels))
    return images[order], labels[order]


images, labels = build_sample_images()

# 70% training, 15% validation, 15% test
train_end = int(len(labels) * 0.7)
validation_end = int(len(labels) * 0.85)

x_train, y_train = images[:train_end], labels[:train_end]
x_val, y_val = images[train_end:validation_end], labels[train_end:validation_end]
x_test, y_test = images[validation_end:], labels[validation_end:]


def prepare_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    return image, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)
    .map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

base = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
]

print("Phase 1: Train the classifier head")

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10,
    callbacks=callbacks,
)

print("Phase 2: Fine-tune the top backbone layers")

base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10,
    callbacks=callbacks,
)

loss, accuracy = model.evaluate(test_ds, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)

with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    file.write("\n".join(class_names))


def classify_property_image(image_path):
    image = keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )
    image = keras.utils.img_to_array(image)
    image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    probabilities = model.predict(image, verbose=0)[0]
    best_index = probabilities.argmax()

    return {
        "category": class_names[best_index],
        "confidence": float(probabilities[best_index]),
    }


# Example:
# result = classify_property_image("sample_bedroom.jpg")
# print(result)

print(
    "\nThese results use synthetic images. "
    "They are for learning, not real property photo classification."
)
