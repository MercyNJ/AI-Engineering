
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. Create and display an image

image = np.zeros((300, 400, 3), dtype=np.uint8)

cv2.rectangle(image, (50, 50), (200, 200), (255, 255, 255), -1)
cv2.circle(image, (300, 120), 50, (0, 165, 255), -1)

cv2.putText(
    image,
    "OpenCV Lab",
    (100, 270),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.imwrite("generated_image.jpg", image)

loaded_image = cv2.imread("generated_image.jpg")

if loaded_image is not None:
    rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_image)
    plt.title("Generated Image")
    plt.axis("off")
    plt.show()


# 2. Resize an image

resized_image = cv2.resize(image, (200, 150))

plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.axis("off")
plt.show()


# Resize while maintaining the original aspect ratio

def resize_pad(image, target_width, target_height):
    height, width = image.shape[:2]

    scale = min(target_width / width, target_height / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(image, (new_width, new_height))

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


padded_image = resize_pad(image, 300, 300)

plt.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
plt.title("Resized with Padding")
plt.axis("off")
plt.show()


# 3. Crop an image

cropped_image = image[40:210, 40:220]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("Cropped Image")
plt.axis("off")
plt.show()


# 4. Colour spaces

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(rgb_image)
axes[0].set_title("RGB")
axes[1].imshow(gray_image, cmap="gray")
axes[1].set_title("Grayscale")
axes[2].imshow(hsv_image)
axes[2].set_title("HSV")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


# Detect orange areas using HSV

lower_orange = np.array([5, 100, 100])
upper_orange = np.array([25, 255, 255])

orange_mask = cv2.inRange(
    hsv_image,
    lower_orange,
    upper_orange
)

plt.imshow(orange_mask, cmap="gray")
plt.title("Orange Colour Mask")
plt.axis("off")
plt.show()


# 5. Blur and edge detection

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

edges = cv2.Canny(blurred_image, 50, 150)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
axes[1].set_title("Blurred")
axes[2].imshow(edges, cmap="gray")
axes[2].set_title("Canny Edges")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


# 6. Drawing shapes and adding text

annotated_image = image.copy()

cv2.rectangle(
    annotated_image,
    (40, 40),
    (210, 210),
    (0, 255, 0),
    3
)

cv2.circle(
    annotated_image,
    (300, 120),
    55,
    (255, 0, 0),
    3
)

cv2.putText(
    annotated_image,
    "Detected Object",
    (70, 240),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title("Annotated Image")
plt.axis("off")
plt.show()


# 7. Histogram equalisation

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

equalized = cv2.equalizeHist(gray)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Grayscale")
axes[1].imshow(equalized, cmap="gray")
axes[1].set_title("Histogram Equalisation")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


# CLAHE improves contrast in smaller image regions

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(gray)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Grayscale")
axes[1].imshow(clahe_image, cmap="gray")
axes[1].set_title("CLAHE")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


# 8. Complete computer vision exercise

canvas = np.zeros((200, 200, 3), dtype=np.uint8)

cv2.rectangle(
    canvas,
    (10, 10),
    (90, 90),
    (255, 255, 255),
    -1
)

cv2.circle(
    canvas,
    (100, 100),
    40,
    (0, 165, 255),
    -1
)

cv2.putText(
    canvas,
    "CV Lab",
    (50, 185),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
)

gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0].set_title("Colour Canvas")
axes[0].axis("off")

axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Canny Edges")
axes[1].axis("off")

plt.tight_layout()
plt.show()