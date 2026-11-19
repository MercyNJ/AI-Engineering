
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# 1. Create a sample property image

def create_property_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    image[:] = (210, 200, 160)

    cv2.rectangle(
        image,
        (0, 320),
        (640, 480),
        (70, 140, 70),
        -1
    )

    cv2.rectangle(
        image,
        (140, 160),
        (500, 360),
        (180, 180, 180),
        -1
    )

    roof = np.array([
        [100, 165],
        [320, 45],
        [540, 165]
    ])

    cv2.fillPoly(
        image,
        [roof],
        (50, 50, 150)
    )

    cv2.rectangle(
        image,
        (270, 230),
        (360, 360),
        (60, 90, 130),
        -1
    )

    cv2.rectangle(
        image,
        (175, 195),
        (245, 265),
        (200, 220, 150),
        -1
    )

    cv2.rectangle(
        image,
        (395, 195),
        (465, 265),
        (200, 220, 150),
        -1
    )

    return image


property_image = create_property_image()

plt.imshow(cv2.cvtColor(property_image, cv2.COLOR_BGR2RGB))
plt.title("Sample Property")
plt.axis("off")
plt.show()


# 2. Load a pretrained YOLO model

model = YOLO("yolov8n.pt")


# 3. Detect objects in the property image

results = model.predict(
    source=property_image,
    conf=0.2,
    iou=0.45,
    verbose=False
)

for result in results:
    print(f"\nDetected {len(result.boxes)} objects")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        class_name = result.names[class_id]

        print(
            f"{class_name}: {confidence:.0%} "
            f"at {[int(value) for value in coordinates]}"
        )


# 4. Display automatic detections

annotated_property = results[0].plot()

plt.figure(figsize=(10, 7))
plt.imshow(
    cv2.cvtColor(annotated_property, cv2.COLOR_BGR2RGB)
)
plt.title("Property Object Detection")
plt.axis("off")
plt.show()


# 5. Manually draw detections

manual_property = property_image.copy()

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = [
            int(value) for value in box.xyxy[0]
        ]

        label = f"{result.names[class_id]} {confidence:.0%}"

        cv2.rectangle(
            manual_property,
            (x1, y1),
            (x2, y2),
            (0, 200, 100),
            2
        )

        label_y = max(y1 - 8, 20)

        cv2.putText(
            manual_property,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 100),
            2
        )

plt.imshow(
    cv2.cvtColor(manual_property, cv2.COLOR_BGR2RGB)
)
plt.title("Manually Annotated Property")
plt.axis("off")
plt.show()


# 6. Compare confidence and IoU thresholds

strict_results = model.predict(
    source=property_image,
    conf=0.6,
    iou=0.3,
    verbose=False
)

loose_results = model.predict(
    source=property_image,
    conf=0.2,
    iou=0.6,
    verbose=False
)

print("\nProperty Detection Comparison")

print(
    f"Strict settings: "
    f"{len(strict_results[0].boxes)} objects"
)

print(
    f"Loose settings: "
    f"{len(loose_results[0].boxes)} objects"
)


# 7. Process a property video

video_path = "sample_property_video.avi"

height, width = property_image.shape[:2]

video_writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*"MJPG"),
    10,
    (width, height)
)

for frame_number in range(10):
    frame = property_image.copy()

    cv2.putText(
        frame,
        f"Property inspection frame {frame_number + 1}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    video_writer.write(frame)

video_writer.release()


# Run YOLO on each video frame

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    video_results = model.predict(
        source=frame,
        conf=0.2,
        verbose=False
    )

    annotated_frame = video_results[0].plot()

    cv2.imshow("Property Video Detection", annotated_frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# 8. Fine-tune YOLO for property-specific classes

# Example classes:
# 0: door
# 1: window
# 2: roof_damage
# 3: wall_crack

def train_property_detector(dataset_yaml):
    property_model = YOLO("yolov8n.pt")

    training_results = property_model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/property_detection",
        name="property_detector"
    )

    return training_results


# Example:
# train_property_detector("property_dataset.yaml")


# 9. Use the trained property detector

def detect_property_features(image_path, trained_model_path):
    property_model = YOLO(trained_model_path)

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Could not load property image.")

    results = property_model.predict(
        source=image,
        conf=0.4,
        iou=0.45,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            print(f"{class_name}: {confidence:.0%}")

    return results[0].plot()


# Example after training:
#
# detected_property = detect_property_features(
#     "property_photo.jpg",
#     "runs/property_detection/property_detector/weights/best.pt"
# )
#
# plt.imshow(
#     cv2.cvtColor(detected_property, cv2.COLOR_BGR2RGB)
# )
# plt.axis("off")
# plt.show()


# 10. Architecture task comparison

print("\nProperty Computer Vision Applications")

print("Classification: Identify the general property type.")
print("Detection: Locate doors, windows, people and vehicles.")
print("Segmentation: Identify exact pixels belonging to property features.")
print("Custom YOLO: Detect specialised property features after training.")
