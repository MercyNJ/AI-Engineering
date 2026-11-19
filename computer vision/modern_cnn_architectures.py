
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# 1. A residual block with a skip connection

def residual_block(x, filters, stride=1):
    shortcut = x

    x = keras.layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False
        )(shortcut)

        shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.ReLU()(x)

    return x


# Build a small CNN using residual blocks

inputs = keras.Input(shape=(224, 224, 3))

x = keras.layers.Conv2D(
    64,
    7,
    strides=2,
    padding="same"
)(inputs)

x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)

x = residual_block(x, 64)
x = residual_block(x, 128, stride=2)
x = residual_block(x, 256, stride=2)

x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

residual_model = keras.Model(inputs, outputs)

print("\nCustom Residual CNN")
residual_model.summary()


# 2. Generate a sample image

sample_image = np.zeros((224, 224, 3), dtype=np.uint8)

cv2_rectangle = (40, 40, 180, 180)

import cv2

cv2.rectangle(
    sample_image,
    (cv2_rectangle[0], cv2_rectangle[1]),
    (cv2_rectangle[2], cv2_rectangle[3]),
    (180, 120, 70),
    -1
)

cv2.circle(
    sample_image,
    (110, 110),
    35,
    (0, 165, 255),
    -1
)

plt.imshow(sample_image)
plt.title("Sample Image")
plt.axis("off")
plt.show()


# 3. Load pretrained ResNet50

# Set to False if you do not want to download ImageNet weights.
LOAD_IMAGENET_WEIGHTS = True

weights = "imagenet" if LOAD_IMAGENET_WEIGHTS else None

resnet = keras.applications.ResNet50(
    include_top=False,
    weights=weights,
    input_shape=(224, 224, 3)
)

resnet.trainable = False

resnet_inputs = keras.Input(shape=(224, 224, 3))

x = keras.applications.resnet.preprocess_input(resnet_inputs)
x = resnet(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)

resnet_outputs = keras.layers.Dense(
    5,
    activation="softmax"
)(x)

resnet_classifier = keras.Model(
    resnet_inputs,
    resnet_outputs
)

print("\nResNet50 Transfer Learning Model")
print(f"Parameters: {resnet_classifier.count_params():,}")

resnet_classifier.summary()


# 4. Use ResNet50 to extract image features

resnet_input = np.expand_dims(
    sample_image.astype("float32"),
    axis=0
)

resnet_input = keras.applications.resnet.preprocess_input(
    resnet_input
)

features = resnet.predict(resnet_input, verbose=0)

print("\nResNet50 feature shape:")
print(features.shape)


# 5. Load pretrained EfficientNetB0

efficientnet = keras.applications.EfficientNetB0(
    include_top=False,
    weights=weights,
    input_shape=(224, 224, 3)
)

efficientnet.trainable = False

efficientnet_inputs = keras.Input(shape=(224, 224, 3))

# EfficientNet includes its own input rescaling.
x = efficientnet(efficientnet_inputs, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)

efficientnet_outputs = keras.layers.Dense(
    5,
    activation="softmax"
)(x)

efficientnet_classifier = keras.Model(
    efficientnet_inputs,
    efficientnet_outputs
)

print("\nEfficientNetB0 Transfer Learning Model")
print(f"Parameters: {efficientnet_classifier.count_params():,}")

efficientnet_classifier.summary()


# Extract features using EfficientNetB0

efficientnet_input = np.expand_dims(
    sample_image.astype("float32"),
    axis=0
)

efficientnet_features = efficientnet.predict(
    efficientnet_input,
    verbose=0
)

print("\nEfficientNetB0 feature shape:")
print(efficientnet_features.shape)


# 6. Compare model parameters

vgg16 = keras.applications.VGG16(
    include_top=False,
    weights=None,
    input_shape=(224, 224, 3)
)

model_comparison = {
    "VGG16": vgg16.count_params(),
    "ResNet50": resnet.count_params(),
    "EfficientNetB0": efficientnet.count_params()
}

print("\nArchitecture Parameter Comparison")

for model_name, parameters in model_comparison.items():
    print(f"{model_name}: {parameters:,}")


# 7. Vision Transformer using Hugging Face

def try_vision_transformer(image):
    try:
        import torch
        from transformers import (
            ViTImageProcessor,
            ViTForImageClassification
        )

        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

        vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = vit(**inputs)

        probabilities = torch.nn.functional.softmax(
            outputs.logits,
            dim=-1
        )

        predicted_class = probabilities.argmax(-1).item()

        print("\nVision Transformer Prediction:")
        print(vit.config.id2label[predicted_class])

    except ImportError:
        print(
            "\nInstall transformers and torch to use "
            "the Vision Transformer example."
        )


# Uncomment to load the pretrained Vision Transformer.
# try_vision_transformer(sample_image)


# 8. Architecture selection

print("\nArchitecture Guide")

print("MobileNetV2: Mobile and edge applications")
print("EfficientNetB0: Efficient image recognition")
print("ResNet50: Balanced deep CNN")
print("EfficientNetB4-B7: Higher accuracy with more resources")
print("Vision Transformer: Image recognition using attention")
