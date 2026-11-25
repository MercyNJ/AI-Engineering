
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import astronaut


# 1. Load a built-in sample image

image = astronaut()

image_bgr = cv2.cvtColor(
    image,
    cv2.COLOR_RGB2BGR
)

print("Image shape:", image.shape)


# 2. Face detection using OpenCV Haar Cascades

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

gray_image = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2GRAY
)

faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

detected_image = image.copy()

print("\nNumber of detected faces:", len(faces))

for (x, y, width, height) in faces:

    cv2.rectangle(
        detected_image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )


plt.figure(figsize=(8, 6))
plt.imshow(detected_image)
plt.title("Face Detection")
plt.axis("off")
plt.show()


# 3. Generate face embeddings using face_recognition

# The built-in sample provides a face image without requiring an external file.

known_image = image.copy()

# Create a slightly modified version of the same image

unknown_image = cv2.convertScaleAbs(
    image,
    alpha=0.9,
    beta=10
)

known_encodings = face_recognition.face_encodings(
    known_image
)

unknown_encodings = face_recognition.face_encodings(
    unknown_image
)

if not known_encodings or not unknown_encodings:
    raise ValueError(
        "Could not generate face embeddings from the sample images."
    )

known_encoding = known_encodings[0]

print("\nKnown face embedding shape:", known_encoding.shape)


# 4. Compare face embeddings

for encoding in unknown_encodings:

    distance = face_recognition.face_distance(
        [known_encoding],
        encoding
    )[0]

    tolerance = 0.6

    match = distance <= tolerance

    print("Face distance:", round(float(distance), 4))
    print("Match:", bool(match))


# 5. Compare different tolerance thresholds

distances = face_recognition.face_distance(
    [known_encoding],
    unknown_encodings[0]
)

print("\nTolerance comparison:")

for tolerance in [0.4, 0.5, 0.6, 0.7]:

    match = distances[0] <= tolerance

    print(
        f"Tolerance {tolerance}:",
        "Match" if match else "No match"
    )


# 6. Visualise the known and comparison images

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(known_image)
plt.title("Reference Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(unknown_image)
plt.title("Comparison Image")
plt.axis("off")

plt.tight_layout()
plt.show()
