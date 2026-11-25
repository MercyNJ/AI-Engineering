
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

print("MobileNetV2 model loaded.")


# 2. Define frame prediction

def predict_frame(frame):

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

    result = decode(predictions, top=1)[0][0]

    return result


# 3. Open the webcam

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check your camera connection.")

print("Requested FPS:", cap.get(cv2.CAP_PROP_FPS))


# 4. Process live video frames

frame_count = 0
prev_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imshow("Grayscale Video", gray)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Processed frames:", frame_count)


# 5. Run MobileNetV2 on live webcam frames

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not reopen webcam.")

frame_count = 0
prev_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    label, name, probability = predict_frame(frame)

    current_time = time.time()
    fps = 1 / max(current_time - prev_time, 1e-7)
    prev_time = current_time

    cv2.putText(
        frame,
        f"{name}: {probability:.0%}  FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Classification", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Classified frames:", frame_count)


# 6. Optimise processing by predicting every fifth frame

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not reopen webcam.")

frame_count = 0
prediction = ("Unknown", 0.0)
prev_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % 5 == 0:

        label, name, probability = predict_frame(frame)

        prediction = (name, probability)

    current_time = time.time()
    fps = 1 / max(current_time - prev_time, 1e-7)
    prev_time = current_time

    cv2.putText(
        frame,
        f"{prediction[0]}: {prediction[1]:.0%}  FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Optimised Live Classification", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# 7. Convert MobileNetV2 to TensorFlow Lite

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

tflite_path = Path("mobilenet_v2.tflite")

tflite_path.write_bytes(tflite_model)

print("\nTensorFlow Lite model saved:", tflite_path)


# 8. Load the TensorFlow Lite model

interpreter = tf.lite.Interpreter(
    model_path=str(tflite_path)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# 9. Define TensorFlow Lite prediction

def predict_tflite(frame):

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

    return predictions


# 10. Test TensorFlow Lite using the webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not reopen webcam.")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    predictions = predict_tflite(frame)

    result = decode(predictions, top=1)[0][0]

    _, name, probability = result

    cv2.putText(
        frame,
        f"TFLite: {name} {probability:.0%}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("TensorFlow Lite Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# 11. Optional ONNX Runtime conversion and inference

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

    onnx_path = Path("mobilenet_v2.onnx")
    onnx_path.write_bytes(onnx_model.SerializeToString())

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    def predict_onnx(frame):

        image = cv2.resize(frame, (224, 224))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = preprocess(
            image[np.newaxis].astype("float32")
        )

        input_name = session.get_inputs()[0].name

        return session.run(
            None,
            {input_name: image}
        )[0]

    print("\nONNX model saved and loaded successfully.")

except ImportError:

    print(
        "\nONNX Runtime demonstration skipped. "
        "Install tf2onnx and onnxruntime to enable it."
    )

except Exception as error:

    print("\nONNX conversion was unsuccessful:", error)
