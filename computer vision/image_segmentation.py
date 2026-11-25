
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# 1. Generate a synthetic image segmentation dataset

np.random.seed(42)

IMAGE_SIZE = 64
NUM_CLASSES = 3


def generate_image():
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]

    # Generate a circle

    center_x = np.random.randint(15, 49)
    center_y = np.random.randint(15, 49)
    radius = np.random.randint(7, 14)

    circle = (
        (x - center_x) ** 2 +
        (y - center_y) ** 2
    ) <= radius ** 2

    mask[circle] = 1

    # Generate a rectangle

    x1 = np.random.randint(2, 35)
    y1 = np.random.randint(2, 35)
    width = np.random.randint(12, 25)
    height = np.random.randint(12, 25)

    rectangle = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    rectangle[y1:y1 + height, x1:x1 + width] = True

    mask[rectangle] = 2

    # Give each class a different colour

    image[mask == 1] = [1.0, 0.2, 0.2]
    image[mask == 2] = [0.2, 0.8, 0.2]

    # Add a small amount of background variation

    image += np.random.normal(
        0, 0.03, image.shape
    ).astype(np.float32)

    image = np.clip(image, 0, 1)

    return image, mask


number_of_images = 300

generated = [
    generate_image()
    for _ in range(number_of_images)
]

images = np.array([item[0] for item in generated])
masks = np.array([item[1] for item in generated])

split_index = int(number_of_images * 0.8)

X_train = images[:split_index]
y_train = masks[:split_index]

X_test = images[split_index:]
y_test = masks[split_index:]

print("Training images:", X_train.shape)
print("Training masks:", y_train.shape)
print("Testing images:", X_test.shape)


# 2. Visualise sample images and their segmentation masks

class_names = ["Background", "Circle", "Rectangle"]

plt.figure(figsize=(10, 5))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_train[i])
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(y_train[i], vmin=0, vmax=2, cmap="viridis")
    plt.title("Segmentation Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()


# 3. Build a U-Net model

def conv_block(x, filters):

    x = keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu"
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

print("\nIoU scores:")

for name, score in zip(class_names, iou_scores):
    print(f"{name}: {score:.4f}")

print("Mean IoU:", round(np.mean(iou_scores), 4))


# 8. Visualise segmentation predictions

plt.figure(figsize=(12, 9))

for i in range(4):

    predicted_mask = np.argmax(
        predictions[i],
        axis=-1
    )

    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(X_test[i])
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(y_test[i], vmin=0, vmax=2, cmap="viridis")
    plt.title("Actual Mask")
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(predicted_mask, vmin=0, vmax=2, cmap="viridis")
    plt.title("Predicted Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()


# 9. Pretrained segmentation model using DeepLabV3

import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

from PIL import Image


weights = DeepLabV3_ResNet50_Weights.DEFAULT

pretrained_model = deeplabv3_resnet50(
    weights=weights
)

pretrained_model.eval()

# Use a generated image instead of requiring an external image file

sample_image = (
    X_test[0] * 255
).astype(np.uint8)

pil_image = Image.fromarray(sample_image)

preprocess = weights.transforms()

input_tensor = preprocess(
    pil_image
).unsqueeze(0)

with torch.no_grad():

    output = pretrained_model(input_tensor)["out"][0]

    pretrained_mask = output.argmax(0).numpy()


print("\nPretrained model output shape:", pretrained_mask.shape)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(pil_image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pretrained_mask, cmap="viridis")
plt.title("DeepLabV3 Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
