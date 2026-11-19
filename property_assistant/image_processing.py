
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. Create a sample property image

property_image = np.zeros((400, 500, 3), dtype=np.uint8)

property_image[:] = (220, 200, 150)

cv2.rectangle(
    property_image,
    (0, 280),
    (500, 400),
    (70, 150, 70),
    -1
)

cv2.rectangle(
    property_image,
    (100, 140),
    (400, 320),
    (180, 180, 180),
    -1
)

roof_points = np.array([
    [70, 150],
    [250, 40],
    [430, 150]
])

cv2.fillPoly(
    property_image,
    [roof_points],
    (50, 50, 150)
)

cv2.rectangle(
    property_image,
    (210, 220),
    (290, 320),
    (60, 90, 130),
    -1
)

cv2.rectangle(
    property_image,
    (130, 175),
    (190, 230),
    (200, 220, 150),
    -1
)

cv2.rectangle(
    property_image,
    (310, 175),
    (370, 230),
    (200, 220, 150),
    -1
)

cv2.imwrite("sample_property.jpg", property_image)

loaded_image = cv2.imread("sample_property.jpg")

if loaded_image is not None:
    plt.imshow(cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB))
    plt.title("Sample Property")
    plt.axis("off")
    plt.show()


# 2. Resize a property image

resized_property = cv2.resize(property_image, (250, 200))

plt.imshow(cv2.cvtColor(resized_property, cv2.COLOR_BGR2RGB))
plt.title("Resized Property Image")
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


padded_property = resize_pad(property_image, 400, 400)

plt.imshow(cv2.cvtColor(padded_property, cv2.COLOR_BGR2RGB))
plt.title("Property Image with Padding")
plt.axis("off")
plt.show()


# 3. Crop a section of the property image

front_door = property_image[210:330, 200:300]

plt.imshow(cv2.cvtColor(front_door, cv2.COLOR_BGR2RGB))
plt.title("Cropped Front Door")
plt.axis("off")
plt.show()


# 4. Colour spaces

gray_property = cv2.cvtColor(
    property_image,
    cv2.COLOR_BGR2GRAY
)

rgb_property = cv2.cvtColor(
    property_image,
    cv2.COLOR_BGR2RGB
)

hsv_property = cv2.cvtColor(
    property_image,
    cv2.COLOR_BGR2HSV
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(rgb_property)
axes[0].set_title("RGB")
axes[1].imshow(gray_property, cmap="gray")
axes[1].set_title("Grayscale")
axes[2].imshow(hsv_property)
axes[2].set_title("HSV")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


# Detect green areas such as gardens

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

garden_mask = cv2.inRange(
    hsv_property,
    lower_green,
    upper_green
)

plt.imshow(garden_mask, cmap="gray")
plt.title("Green Area Mask")
plt.axis("off")
plt.show()


# 5. Blur and edge detection

blurred_property = cv2.GaussianBlur(
    property_image,
    (5, 5),
    0
)

property_edges = cv2.Canny(
    blurred_property,
    50,
    150
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(cv2.cvtColor(property_image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Property")

axes[1].imshow(cv2.cvtColor(blurred_property, cv2.COLOR_BGR2RGB))
axes[1].set_title("Blurred Property")

axes[2].imshow(property_edges, cmap="gray")
axes[2].set_title("Property Edges")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


# 6. Annotate property features

annotated_property = property_image.copy()

cv2.rectangle(
    annotated_property,
    (125, 170),
    (195, 235),
    (0, 255, 0),
    3
)

cv2.rectangle(
    annotated_property,
    (305, 170),
    (375, 235),
    (0, 255, 0),
    3
)

cv2.rectangle(
    annotated_property,
    (205, 215),
    (295, 325),
    (255, 0, 0),
    3
)

cv2.putText(
    annotated_property,
    "Window",
    (115, 160),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    2
)

cv2.putText(
    annotated_property,
    "Door",
    (220, 210),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    2
)

plt.imshow(cv2.cvtColor(annotated_property, cv2.COLOR_BGR2RGB))
plt.title("Annotated Property Features")
plt.axis("off")
plt.show()


# 7. Histogram equalisation

gray = cv2.cvtColor(
    property_image,
    cv2.COLOR_BGR2GRAY
)

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

clahe_property = clahe.apply(gray)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Grayscale")

axes[1].imshow(clahe_property, cmap="gray")
axes[1].set_title("CLAHE")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


# 8. Complete property image processing exercise

property_canvas = np.zeros((300, 350, 3), dtype=np.uint8)

property_canvas[:] = (180, 200, 220)

cv2.rectangle(
    property_canvas,
    (40, 80),
    (310, 260),
    (150, 150, 150),
    -1
)

cv2.rectangle(
    property_canvas,
    (130, 170),
    (210, 260),
    (60, 90, 130),
    -1
)

cv2.circle(
    property_canvas,
    (270, 120),
    25,
    (0, 165, 255),
    -1
)

cv2.putText(
    property_canvas,
    "Property",
    (105, 290),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

gray = cv2.cvtColor(
    property_canvas,
    cv2.COLOR_BGR2GRAY
)

edges = cv2.Canny(gray, 50, 150)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(cv2.cvtColor(property_canvas, cv2.COLOR_BGR2RGB))
axes[0].set_title("Property Canvas")
axes[0].axis("off")

axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Property Edges")
axes[1].axis("off")

plt.tight_layout()
plt.show()