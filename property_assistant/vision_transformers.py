
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification


# 1. Generate a sample property image

image = Image.new(
    "RGB",
    (224, 224),
    color=(225, 220, 205)
)

draw = ImageDraw.Draw(image)

# Draw a window

draw.rectangle(
    [20, 30, 95, 120],
    fill=(130, 180, 210),
    outline=(255, 255, 255),
    width=5
)

# Draw a sofa

draw.rectangle(
    [45, 145, 185, 195],
    fill=(120, 85, 65)
)

draw.rectangle(
    [55, 130, 175, 155],
    fill=(150, 110, 85)
)

# Draw a table

draw.rectangle(
    [95, 195, 150, 205],
    fill=(100, 65, 40)
)

# Draw a floor

draw.rectangle(
    [0, 205, 224, 224],
    fill=(170, 155, 135)
)

print("Sample property image created.")


# 2. Load a pretrained Vision Transformer

model_name = "google/vit-base-patch16-224"

classifier = pipeline(
    "image-classification",
    model=model_name
)

print("Vision Transformer loaded successfully.")


# 3. Classify the property image

results = classifier(image)

print("\nProperty image classification results:")

for result in results[:5]:

    print(
        f"{result['label']}: "
        f"{result['score']:.2%}"
    )


# 4. Visualise the property image

plt.figure(figsize=(6, 5))

plt.imshow(image)
plt.title("Sample Property Interior")
plt.axis("off")
plt.show()


# 5. Divide the property image into patches

image_size = 224
patch_size = 16

image_array = np.array(
    image.resize((image_size, image_size))
)

patches = []

for y in range(0, image_size, patch_size):
    for x in range(0, image_size, patch_size):

        patch = image_array[
            y:y + patch_size,
            x:x + patch_size
        ]

        patches.append(patch)

print("\nProperty image dimensions:", image_array.shape)
print("Patch dimensions:", patches[0].shape)
print("Number of patches:", len(patches))


# 6. Visualise property image patches

plt.figure(figsize=(8, 8))

for i, patch in enumerate(patches[:49]):

    plt.subplot(7, 7, i + 1)
    plt.imshow(patch)
    plt.axis("off")

plt.suptitle("Property Image Divided Into Patches")
plt.tight_layout()
plt.show()


# 7. Compare ViT and CNN predictions

cnn_classifier = pipeline(
    "image-classification",
    model="microsoft/resnet-50"
)

vit_results = classifier(image)
cnn_results = cnn_classifier(image)

print("\nVision Transformer predictions:")

for result in vit_results[:3]:

    print(
        f"{result['label']}: "
        f"{result['score']:.2%}"
    )

print("\nCNN (ResNet-50) predictions:")

for result in cnn_results[:3]:

    print(
        f"{result['label']}: "
        f"{result['score']:.2%}"
    )


# 8. Load the Vision Transformer model

processor = AutoImageProcessor.from_pretrained(model_name)

vit_model = AutoModelForImageClassification.from_pretrained(
    model_name
)

print("\nVision Transformer model loaded.")


# 9. Convert the property image into model inputs

inputs = processor(
    images=image,
    return_tensors="pt"
)

print("\nProcessed property image tensor shape:")
print(inputs["pixel_values"].shape)


# 10. Run direct model inference

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

print("\nProperty image predictions:")

for probability, index in zip(
    top_probabilities[0],
    top_indices[0]
):

    label = vit_model.config.id2label[index.item()]

    print(
        f"{label}: {probability.item():.2%}"
    )
