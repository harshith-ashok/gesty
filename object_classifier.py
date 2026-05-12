import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image

MODEL_DIR = Path("custom_model")
MODEL_PATH = MODEL_DIR / "model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"

IMG_SIZE = 224


class ObjectClassifier:
    def __init__(self):
        self.custom_model = None
        self.class_names = []

        if MODEL_PATH.exists() and CLASS_NAMES_PATH.exists():
            print("Loading custom trained model...")
            self.custom_model = tf.keras.models.load_model(MODEL_PATH)

            with open(CLASS_NAMES_PATH, "r") as f:
                self.class_names = [line.strip() for line in f if line.strip()]

            print(f"Loaded custom classes: {self.class_names}")
        else:
            print("No custom model found. Using ImageNet MobileNetV2.")
            self.custom_model = MobileNetV2(weights="imagenet")

    def preprocess(self, frame):
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = np.expand_dims(rgb.astype(np.float32), axis=0)
        return preprocess_input(arr)

    def predict(self, frame):
        x = self.preprocess(frame)
        preds = self.custom_model.predict(x, verbose=0)

        if self.class_names:
            idx = int(np.argmax(preds[0]))
            confidence = float(preds[0][idx])
            label = self.class_names[idx]
            return label, confidence
        else:
            decoded = decode_predictions(preds, top=1)[0][0]
            label = decoded[1].replace("_", " ")
            confidence = float(decoded[2])
            return label, confidence


def main():
    classifier = ObjectClassifier()
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        label, confidence = classifier.predict(frame)

        cv2.putText(
            frame,
            f"{label} ({confidence * 100:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Object Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
