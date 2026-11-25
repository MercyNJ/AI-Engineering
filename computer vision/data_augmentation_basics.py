
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import albumentations as A
import cv2


# 1. Load the CIFAR-10 dataset

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)
print("Example class:", class_names[int(y_train[0][0])])


# Use a smaller subset to keep this demonstration manageable

X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]


# 2. Basic data augmentation using Keras

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.2),
    keras.layers.RandomTranslation(0.1, 0.1)
])

sample_image = X_train[0]

plt.figure(figsize=(12, 4))

for i in range(10):
    augmented_image = augment(
        tf.expand_dims(sample_image, axis=0),
        training=True
    )

    plt.subplot(2, 5, i + 1)
    plt.imshow(tf.cast(augmented_image[0], tf.uint8))
    plt.axis("off")

plt.suptitle("Different Augmented Versions of One Image")
plt.tight_layout()
plt.show()


# 3. Compare models with and without data augmentation

def make_cnn(with_augmentation=False):

    layers = [
        keras.Input(shape=(32, 32, 3))
    ]

    if with_augmentation:
        layers += [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1)
        ]

    layers += [
        keras.layers.Rescaling(1.0 / 255),

        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(10, activation="softmax")
    ]

    return keras.Sequential(layers)


baseline = make_cnn(with_augmentation=False)
augmented = make_cnn(with_augmentation=True)


# 4. Compile and train both models

for model in [baseline, augmented]:
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

print("\nTraining baseline model...")

history_baseline = baseline.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

print("\nTraining augmented model...")

history_augmented = augmented.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.1,
    verbose=1
)


# 5. Compare validation accuracy

plt.figure(figsize=(9, 5))

plt.plot(
    history_baseline.history["val_accuracy"],
    label="Without augmentation"
)

plt.plot(
    history_augmented.history["val_accuracy"],
    label="With augmentation"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Effect of Data Augmentation on Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# 6. Evaluate both models on the test dataset

baseline_loss, baseline_accuracy = baseline.evaluate(
    X_test, y_test, verbose=0
)

augmented_loss, augmented_accuracy = augmented.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nBaseline test accuracy: {baseline_accuracy:.3f}")
print(f"Augmented test accuracy: {augmented_accuracy:.3f}")


# 7. Advanced augmentation using Albumentations

sample_image = X_train[0]

albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.ColorJitter(p=0.3),
    A.ShiftScaleRotate(p=0.5),
    A.ElasticTransform(p=0.2),
    A.Normalize()
])

transformed = albumentations_transform(image=sample_image)

augmented_image = transformed["image"]

print("\nAlbumentations output shape:", augmented_image.shape)
print("Albumentations output dtype:", augmented_image.dtype)


# 8. MixUp augmentation

def mixup(images, labels, alpha=0.2):

    lam = np.random.beta(alpha, alpha)

    indices = tf.random.shuffle(
        tf.range(tf.shape(images)[0])
    )

    images = tf.cast(images, tf.float32) / 255.0
    labels = tf.cast(labels, tf.float32)

    mixed_images = (
        lam * images +
        (1 - lam) * tf.gather(images, indices)
    )

    mixed_labels = (
        lam * labels +
        (1 - lam) * tf.gather(labels, indices)
    )

    return mixed_images, mixed_labels


# 9. CutMix augmentation

def cutmix(images, labels, alpha=1.0):

    lam = np.random.beta(alpha, alpha)

    batch_size = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]

    indices = tf.random.shuffle(tf.range(batch_size))

    images = tf.cast(images, tf.float32) / 255.0
    labels = tf.cast(labels, tf.float32)

    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    height_value = int(height.numpy())
    width_value = int(width.numpy())

    cut_ratio = np.sqrt(1.0 - lam)

    cut_width = int(width_value * cut_ratio)
    cut_height = int(height_value * cut_ratio)

    center_x = np.random.randint(0, width_value)
    center_y = np.random.randint(0, height_value)

    x1 = np.clip(center_x - cut_width // 2, 0, width_value)
    x2 = np.clip(center_x + cut_width // 2, 0, width_value)

    y1 = np.clip(center_y - cut_height // 2, 0, height_value)
    y2 = np.clip(center_y + cut_height // 2, 0, height_value)

    mask = np.ones(
        (height_value, width_value, 1),
        dtype=np.float32
    )

    mask[y1:y2, x1:x2, :] = 0.0

    mask = tf.constant(mask, dtype=tf.float32)

    mixed_images = (
        images * mask +
        shuffled_images * (1.0 - mask)
    )

    actual_area = (
        (x2 - x1) * (y2 - y1)
    ) / (width_value * height_value)

    adjusted_lam = 1.0 - actual_area

    mixed_labels = (
        adjusted_lam * labels +
        (1.0 - adjusted_lam) * shuffled_labels
    )

    return mixed_images, mixed_labels


# 10. Demonstrate MixUp and CutMix

sample_images = X_train[:8]
sample_labels = keras.utils.to_categorical(
    y_train[:8].flatten(),
    num_classes=10
)

mixed_images, mixed_labels = mixup(
    sample_images,
    sample_labels
)

cutmix_images, cutmix_labels = cutmix(
    sample_images,
    sample_labels
)


# Display MixUp results

plt.figure(figsize=(12, 4))

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(mixed_images[i])
    plt.title("MixUp")
    plt.axis("off")

plt.suptitle("MixUp: Combining Two Images")
plt.tight_layout()
plt.show()


# Display CutMix results

plt.figure(figsize=(12, 4))

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(cutmix_images[i])
    plt.title("CutMix")
    plt.axis("off")

plt.suptitle("CutMix: Pasting One Image Into Another")
plt.tight_layout()
plt.show()


print("\nOriginal labels:")
print(sample_labels[:4])

print("\nMixUp labels:")
print(mixed_labels[:4].numpy())

print("\nCutMix labels:")
print(cutmix_labels[:4].numpy())
