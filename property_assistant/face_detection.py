
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import astronaut


# 1. Generate a sample property-style image

image = np.zeros((512, 512, 3), dtype=np.uint8)

# Create a simple room background

image[:] = [225, 220, 205]

# Floor

image[380:512, :] = [170, 155, 135]

# Window

image[60:240, 40:190] = [130, 180, 210]

# Window frame

image[145:155, 40:190] = [245, 245, 245]
image[60:240, 110:120] = [245, 245, 245]

# Furniture

image[290:380, 240:450] = [120, 85, 65]
image[270:300, 220:450] = [150, 110, 85]

# Add a built-in sample face to simulate a person appearing in a property photo

person_image = cv2.resize(
    astronaut(),
    (110, 110)
)

image[100:210, 300:410] = person_image

print("Sample property image shape:", image.shape)


# 2. Load the face detection model

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# 3. Detect faces in the property image

gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_RGB2GRAY
)

faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(25, 25)
)

print("\nNumber of detected faces:", len(faces))


# 4. Draw bounding boxes around detected faces

detected_image = image.copy()

for (x, y, width, height) in faces:

    cv2.rectangle(
        detected_image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )


# 5. Blur detected faces to protect privacy

blurred_image = image.copy()

for (x, y, width, height) in faces:

    face_region = blurred_image[
        y:y + height,
        x:x + width
    ]

    blurred_image[
        y:y + height,
        x:x + width
    ] = cv2.GaussianBlur(
        face_region,
        (51, 51),
        30
    )


# 6. Visualise the results

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Property Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(detected_image)
plt.title("Detected Faces")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blurred_image)
plt.title("Faces Blurred")
plt.axis("off")

plt.tight_layout()
plt.show()


# 7. Display detection summary

print("\nProperty Image Privacy Summary")
print("--------------------------------")
print("Faces detected:", len(faces))
print("Faces blurred:", len(faces))

if len(faces) > 0:
    print("Privacy processing completed.")
else:
    print("No faces were detected in the image.")
