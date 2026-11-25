
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from torchvision.datasets import CIFAR10
from torchvision import transforms


# 1. Load a pretrained Vision Transformer

model_name = "google/vit-base-patch16-224"

classifier = pipeline(
    "image-classification",
    model=model_name
)

print("Vision Transformer loaded successfully.")


# 2. Load a sample image from CIFAR-10

dataset = CIFAR10(
    root="data",
    train=False,
    download=True
)

image, actual_label = dataset[0]

print("\nActual image class:", dataset.classes[actual_label])


# 3. Classify the image using ViT

results = classifier(image)

print("\nVision Transformer predictions:")

for result in results[:5]:
    print(
        f"{result['label']}: "
        f"{result['score']:.2%}"
    )


# 4. Visualise the image and predictions

plt.figure(figsize=(6, 5))

plt.imshow(image)
plt.title("Input Image")
plt.axis("off")
plt.show()


# 5. Understand image patch extraction

image_size = 224
patch_size = 16

sample_image = image.resize((image_size, image_size))

image_array = np.array(sample_image)

patches = []

for y in range(0, image_size, patch_size):
    for x in range(0, image_size, patch_size):

        patch = image_array[
            y:y + patch_size,
            x:x + patch_size
        ]

        patches.append(patch)

print("\nImage dimensions:", image_array.shape)
print("Patch dimensions:", patches[0].shape)
print("Number of patches:", len(patches))


# 6. Visualise image patches

plt.figure(figsize=(8, 8))

for i, patch in enumerate(patches[:49]):

    plt.subplot(7, 7, i + 1)
    plt.imshow(patch)
    plt.axis("off")

plt.suptitle("Vision Transformer Image Patches")
plt.tight_layout()
plt.show()


# 7. Compare CNN and Vision Transformer predictions

cnn_classifier = pipeline(
    "image-classification",
    model="microsoft/resnet-50"
)

vit_results = classifier(image)
cnn_results = cnn_classifier(image)

print("\nVision Transformer predictions:")

for result in vit_results[:3]:
    print(f"{result['label']}: {result['score']:.2%}")

print("\nCNN (ResNet-50) predictions:")

for result in cnn_results[:3]:
    print(f"{result['label']}: {result['score']:.2%}")


# 8. Load the ViT model and inspect its architecture

processor = AutoImageProcessor.from_pretrained(model_name)

vit_model = AutoModelForImageClassification.from_pretrained(
    model_name
)

print("\nVision Transformer architecture:")
print(vit_model)


# 9. Process an image into model inputs

inputs = processor(
    images=image,
    return_tensors="pt"
)

print("\nProcessed image tensor shape:")
print(inputs["pixel_values"].shape)


# 10. Run inference directly through the model

vit_model.eval()

with torch.no_grad():

    outputs = vit_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3,
        dim=-1
    )

print("\nDirect model predictions:")

for probability, index in zip(
    top_probabilities[0],
    top_indices[0]
):

    label = vit_model.config.id2label[index.item()]

    print(
        f"{label}: {probability.item():.2%}"
    )
