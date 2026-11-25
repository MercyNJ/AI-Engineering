
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from pathlib import Path


# 1. Load a pretrained MobileNetV2 model

model = keras.applications.MobileNetV2(
    weights="imagenet"
)

preprocess = keras.applications.mobilenet_v2.preprocess_input
decode = keras.applications.imagenet_utils.decode_predictions

print("Property image analysis model loaded.")


# 2. Define property image classification

def classify_property_frame(frame):

    image = cv2.resize(frame, (224, 224))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = preprocess(
        image[np.newaxis].astype("float32")
    )

    predictions = model.predict(
        image,
        verbose=0
    )

    result = decode(predictions, top=3)[0]

    return result


# 3. Assess image quality

def assess_image_quality(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if brightness < 60:
        lighting = "Too dark"

    elif brightness > 200:
        lighting = "Too bright"

    else:
        lighting = "Good lighting"

    if sharpness < 50:
        clarity = "Possibly blurry"

    else:
        clarity = "Clear image"

    return brightness, sharpness, lighting, clarity


# 4. Open webcam for property image assessment

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check your camera connection.")


# 5. Process property viewing frames

frame_count = 0
prediction = []
quality = (0, 0, "Unknown", "Unknown")

previous_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Update image analysis every fifth frame

    if frame_count % 5 == 0:

        prediction = classify_property_frame(frame)

        quality = assess_image_quality(frame)

    current_time = time.time()

    fps = 1 / max(
        current_time - previous_time,
        1e-7
    )

    previous_time = current_time

    # Display the most likely image classifications

    if prediction:

        _, label, confidence = prediction[0]

        cv2.putText(
            frame,
            f"Image classification: {label} ({confidence:.0%})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # Display image quality information

    brightness, sharpness, lighting, clarity = quality

    cv2.putText(
        frame,
        f"Lighting: {lighting}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Image: {clarity}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Property Image Assistant",
        frame
    )

    frame_count += 1

    # Press S to save a property image

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        save_directory = Path("property_images")
        save_directory.mkdir(exist_ok=True)

        image_path = save_directory / (
            f"property_image_{frame_count}.jpg"
        )

        cv2.imwrite(
            str(image_path),
            frame
        )

        print("\nProperty image saved:", image_path)

        print("Brightness:", round(float(brightness), 2))
        print("Sharpness:", round(float(sharpness), 2))
        print("Lighting:", lighting)
        print("Clarity:", clarity)

        if prediction:

            print("\nImage classifications:")

            for _, label, confidence in prediction:
                print(f"{label}: {confidence:.2%}")

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\nTotal frames processed:", frame_count)


# 6. Convert MobileNetV2 to TensorFlow Lite

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

tflite_path = Path("property_mobilenet_v2.tflite")

tflite_path.write_bytes(tflite_model)

print("\nTensorFlow Lite model saved:", tflite_path)


# 7. Load the TensorFlow Lite model

interpreter = tf.lite.Interpreter(
    model_path=str(tflite_path)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# 8. Define TensorFlow Lite property image prediction

def predict_property_tflite(frame):

    image = cv2.resize(frame, (224, 224))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = preprocess(
        image[np.newaxis].astype("float32")
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )

    return decode(predictions, top=3)[0]


# 9. Test TensorFlow Lite with the webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not reopen webcam.")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    predictions = predict_property_tflite(frame)

    _, label, confidence = predictions[0]

    cv2.putText(
        frame,
        f"TFLite: {label} ({confidence:.0%})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Property Assistant - TFLite",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# 10. Optional ONNX Runtime conversion and inference

try:

    import tf2onnx
    import onnxruntime as ort

    input_signature = (
        tf.TensorSpec(
            (None, 224, 224, 3),
            tf.float32,
            name="input"
        ),
    )

    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13
    )

    onnx_path = Path("property_mobilenet_v2.onnx")

    onnx_path.write_bytes(
        onnx_model.SerializeToString()
    )

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    def predict_property_onnx(frame):

        image = cv2.resize(frame, (224, 224))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = preprocess(
            image[np.newaxis].astype("float32")
        )

        input_name = session.get_inputs()[0].name

        predictions = session.run(
            None,
            {input_name: image}
        )[0]

        return decode(predictions, top=3)[0]

    print("\nONNX model saved and loaded successfully.")

except ImportError:

    print(
        "\nONNX demonstration skipped. "
        "Install tf2onnx and onnxruntime to enable it."
    )

except Exception as error:

    print("\nONNX conversion was unsuccessful:", error)
