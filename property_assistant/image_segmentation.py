
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# 1. Generate synthetic property floor plans

np.random.seed(42)

IMAGE_SIZE = 64
NUM_CLASSES = 3

class_names = ["Background", "Room", "Wall"]


def generate_floor_plan():

    image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.float32
    )

    mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    # Draw a simple outer wall

    mask[5:59, 5:59] = 2

    # Divide the floor plan into room areas

    image[7:30, 7:30] = [0.7, 0.8, 1.0]
    mask[7:30, 7:30] = 1

    image[7:30, 34:57] = [0.7, 1.0, 0.7]
    mask[7:30, 34:57] = 1

    image[34:57, 7:30] = [1.0, 0.8, 0.6]
    mask[34:57, 7:30] = 1

    image[34:57, 34:57] = [1.0, 0.9, 0.7]
    mask[34:57, 34:57] = 1

    # Add internal walls

    mask[5:59, 31:34] = 2
    mask[30:34, 5:59] = 2

    # Give wall pixels a distinct colour

    image[mask == 2] = [0.2, 0.2, 0.2]

    # Add small variations to make the examples different

    image += np.random.normal(
        0, 0.02, image.shape
    ).astype(np.float32)

    image = np.clip(image, 0, 1)

    return image, mask


number_of_plans = 300

generated_plans = [
    generate_floor_plan()
    for _ in range(number_of_plans)
]

images = np.array([
    item[0] for item in generated_plans
])

masks = np.array([
    item[1] for item in generated_plans
])

split_index = int(number_of_plans * 0.8)

X_train = images[:split_index]
y_train = masks[:split_index]

X_test = images[split_index:]
y_test = masks[split_index:]

print("Training floor plans:", X_train.shape)
print("Training masks:", y_train.shape)
print("Testing floor plans:", X_test.shape)


# 2. Visualise floor plans and their segmentation masks

plt.figure(figsize=(10, 5))

for i in range(4):

    plt.subplot(2, 4, i + 1)
    plt.imshow(X_train[i])
    plt.title("Floor Plan")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(
        y_train[i],
        vmin=0,
        vmax=2,
        cmap="viridis"
    )
    plt.title("Segmentation Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()


# 3. Build a U-Net model

def conv_block(x, filters):

    x = keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = keras.layers.BatchNormalization()(x)

    return x


def build_unet(input_shape=(64, 64, 3), num_classes=3):

    inputs = keras.Input(shape=input_shape)

    skips = []
    x = inputs

    # Encoder

    for filters in [16, 32, 64]:

        x = conv_block(x, filters)
        skips.append(x)

        x = keras.layers.MaxPooling2D()(x)

    # Bottleneck

    x = conv_block(x, 128)

    # Decoder

    for filters, skip in zip(
        [64, 32, 16],
        reversed(skips)
    ):

        x = keras.layers.Conv2DTranspose(
            filters,
            2,
            strides=2,
            padding="same"
        )(x)

        x = keras.layers.Concatenate()([x, skip])

        x = conv_block(x, filters)

    outputs = keras.layers.Conv2D(
        num_classes,
        1,
        activation="softmax"
    )(x)

    return keras.Model(inputs, outputs)


unet = build_unet()

unet.summary()


# 4. Define Dice loss and combined loss

def dice_loss(y_true, y_pred, smooth=1.0):

    y_true = tf.cast(y_true, tf.int32)

    y_true = tf.one_hot(
        y_true,
        depth=NUM_CLASSES
    )

    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1, 2]
    )

    total = (
        tf.reduce_sum(y_true, axis=[1, 2]) +
        tf.reduce_sum(y_pred, axis=[1, 2])
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        total + smooth
    )

    return 1.0 - tf.reduce_mean(dice)


def combined_loss(y_true, y_pred):

    cross_entropy = (
        keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred
        )
    )

    return tf.reduce_mean(cross_entropy) + dice_loss(
        y_true,
        y_pred
    )


unet.compile(
    optimizer="adam",
    loss=combined_loss,
    metrics=["accuracy"]
)


# 5. Train the U-Net model

history = unet.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=8,
    batch_size=16
)


# 6. Evaluate the model

test_loss, test_accuracy = unet.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest loss:", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy, 4))


# 7. Calculate Intersection over Union (IoU)

def compute_iou(y_true, y_pred, num_classes=3):

    y_pred = np.argmax(y_pred, axis=-1)

    iou_scores = []

    for class_id in range(num_classes):

        true_mask = y_true == class_id
        pred_mask = y_pred == class_id

        intersection = np.logical_and(
            true_mask,
            pred_mask
        ).sum()

        union = np.logical_or(
            true_mask,
            pred_mask
        ).sum()

        iou = intersection / (union + 1e-7)

        iou_scores.append(iou)

    return iou_scores


predictions = unet.predict(X_test, verbose=0)

iou_scores = compute_iou(
    y_test,
    predictions,
    NUM_CLASSES
)

print("\nFloor plan segmentation IoU scores:")

for name, score in zip(class_names, iou_scores):
    print(f"{name}: {score:.4f}")

print("Mean IoU:", round(np.mean(iou_scores), 4))


# 8. Visualise predicted floor plan segmentation

plt.figure(figsize=(12, 9))

for i in range(4):

    predicted_mask = np.argmax(
        predictions[i],
        axis=-1
    )

    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(X_test[i])
    plt.title("Original Floor Plan")
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(
        y_test[i],
        vmin=0,
        vmax=2,
        cmap="viridis"
    )
    plt.title("Actual Mask")
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(
        predicted_mask,
        vmin=0,
        vmax=2,
        cmap="viridis"
    )
    plt.title("Predicted Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()
