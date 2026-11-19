
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import ASSETS


# 1. Classification, detection and segmentation

print("Classification: Predicts what an image contains.")
print("Detection: Predicts objects and their bounding boxes.")
print("Segmentation: Identifies the exact pixels belonging to objects.")


# 2. Load a pretrained YOLO model

model = YOLO("yolov8n.pt")


# 3. Load a sample image included with Ultralytics

sample_path = str(ASSETS / "bus.jpg")

image = cv2.imread(sample_path)

if image is None:
    raise FileNotFoundError("Could not load the sample image.")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()


# 4. Detect objects

results = model.predict(
    source=image,
    conf=0.4,
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


# 5. Automatically draw detections

annotated_image = results[0].plot()

plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title("YOLO Object Detections")
plt.axis("off")
plt.show()


# 6. Manually draw bounding boxes and labels

manual_image = image.copy()

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = [
            int(value) for value in box.xyxy[0]
        ]

        label = f"{result.names[class_id]} {confidence:.0%}"

        cv2.rectangle(
            manual_image,
            (x1, y1),
            (x2, y2),
            (0, 200, 100),
            2
        )

        label_y = max(y1 - 8, 20)

        cv2.putText(
            manual_image,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 100),
            2
        )

plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(manual_image, cv2.COLOR_BGR2RGB))
plt.title("Manually Annotated Detections")
plt.axis("off")
plt.show()


# 7. Experiment with confidence and IoU thresholds

strict_results = model.predict(
    source=image,
    conf=0.6,
    iou=0.3,
    verbose=False
)

loose_results = model.predict(
    source=image,
    conf=0.2,
    iou=0.6,
    verbose=False
)

print("\nDetection Comparison")

print(
    f"Strict settings: "
    f"{len(strict_results[0].boxes)} objects"
)

print(
    f"Loose settings: "
    f"{len(loose_results[0].boxes)} objects"
)


# Display the strict and loose results

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(
    cv2.cvtColor(strict_results[0].plot(), cv2.COLOR_BGR2RGB)
)

axes[0].set_title("Strict Detection")

axes[1].imshow(
    cv2.cvtColor(loose_results[0].plot(), cv2.COLOR_BGR2RGB)
)

axes[1].set_title("Loose Detection")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()


# 8. Run YOLO on a generated video

video_path = "sample_detection_video.avi"

height, width = image.shape[:2]

video_writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*"MJPG"),
    10,
    (width, height)
)

for shift in range(0, 50, 5):
    frame = image.copy()

    cv2.putText(
        frame,
        f"Video frame {shift}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    video_writer.write(frame)

video_writer.release()


# Read and process the video frame by frame

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    video_results = model.predict(
        source=frame,
        conf=0.4,
        verbose=False
    )

    annotated_frame = video_results[0].plot()

    cv2.imshow("YOLO Video Detection", annotated_frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# 9. Fine-tune YOLO on a custom dataset

def train_custom_detector(dataset_yaml):
    custom_model = YOLO("yolov8n.pt")

    training_results = custom_model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/custom",
        name="my_detector"
    )

    return training_results


# Example:
# train_custom_detector("custom_dataset.yaml")


# 10. Object detection vs segmentation

print("\nYOLO Task Examples")

print("Detection model: yolov8n.pt")
print("Segmentation model: yolov8n-seg.pt")

# Optional segmentation example:
#
# segmentation_model = YOLO("yolov8n-seg.pt")
# segmentation_results = segmentation_model.predict(image)
#
# plt.imshow(
#     cv2.cvtColor(
#         segmentation_results[0].plot(),
#         cv2.COLOR_BGR2RGB
#     )
# )
# plt.axis("off")
# plt.show()
