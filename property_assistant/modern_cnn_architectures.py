
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2


# 1. Generate a sample property image

def create_property_image():
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    image[:] = (180, 210, 230)

    cv2.rectangle(
        image,
        (0, 160),
        (224, 224),
        (70, 150, 70),
        -1
    )

    cv2.rectangle(
        image,
        (35, 75),
        (190, 180),
        (170, 170, 170),
        -1
    )

    roof = np.array([
        [20, 80],
        [112, 15],
        [205, 80]
    ])

    cv2.fillPoly(
        image,
        [roof],
        (50, 50, 150)
    )

    cv2.rectangle(
        image,
        (90, 115),
        (135, 180),
        (60, 90, 130),
        -1
    )

    cv2.rectangle(
        image,
        (48, 95),
        (82, 130),
        (200, 220, 150),
        -1
    )

    cv2.rectangle(
        image,
        (145, 95),
        (180, 130),
        (200, 220, 150),
        -1
    )

    return image


property_image = create_property_image()

plt.imshow(property_image)
plt.title("Sample Property Image")
plt.axis("off")
plt.show()


# 2. Residual block

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


# Build a custom property image classifier

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

outputs = keras.layers.Dense(
    3,
    activation="softmax"
)(x)

property_residual_model = keras.Model(inputs, outputs)

print("\nCustom Property Residual CNN")
property_residual_model.summary()


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


# Add a property classification head.
# Example categories: apartment, bungalow, maisonette.

resnet_inputs = keras.Input(shape=(224, 224, 3))

x = keras.applications.resnet.preprocess_input(resnet_inputs)

x = resnet(x, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)

x = keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = keras.layers.Dropout(0.3)(x)

resnet_outputs = keras.layers.Dense(
    3,
    activation="softmax"
)(x)

property_resnet = keras.Model(
    resnet_inputs,
    resnet_outputs
)

print("\nProperty ResNet50 Classifier")
print(f"Parameters: {property_resnet.count_params():,}")

property_resnet.summary()


# Extract property image features using ResNet50

property_input = np.expand_dims(
    property_image.astype("float32"),
    axis=0
)

resnet_input = keras.applications.resnet.preprocess_input(
    property_input.copy()
)

property_features = resnet.predict(
    resnet_input,
    verbose=0
)

print("\nProperty ResNet50 Feature Shape:")
print(property_features.shape)


# 4. Load pretrained EfficientNetB0

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

x = keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = keras.layers.Dropout(0.3)(x)

efficientnet_outputs = keras.layers.Dense(
    3,
    activation="softmax"
)(x)

property_efficientnet = keras.Model(
    efficientnet_inputs,
    efficientnet_outputs
)

print("\nProperty EfficientNetB0 Classifier")
print(f"Parameters: {property_efficientnet.count_params():,}")

property_efficientnet.summary()


# Extract property image features using EfficientNetB0

efficientnet_input = np.expand_dims(
    property_image.astype("float32"),
    axis=0
)

efficientnet_features = efficientnet.predict(
    efficientnet_input,
    verbose=0
)

print("\nProperty EfficientNetB0 Feature Shape:")
print(efficientnet_features.shape)


# 5. Compare model parameters

model_comparison = {
    "Custom Residual CNN": property_residual_model.count_params(),
    "ResNet50": resnet.count_params(),
    "EfficientNetB0": efficientnet.count_params()
}

print("\nProperty Model Comparison")

for model_name, parameters in model_comparison.items():
    print(f"{model_name}: {parameters:,}")


# 6. Vision Transformer using Hugging Face

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
# try_vision_transformer(property_image)


# 7. Choosing an architecture for the Property Assistant

print("\nProperty Assistant Architecture Guide")

print("MobileNetV2: Lightweight property image processing")
print("EfficientNetB0: Efficient property image classification")
print("ResNet50: Reusing deep image features")
print("EfficientNetB4-B7: More demanding image classification")
print("Vision Transformer: Learning relationships across image patches")
