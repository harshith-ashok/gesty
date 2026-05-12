# object_classifier.py

from pathlib import Path
from ultralytics import YOLO


class ObjectClassifier:
    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence=0.5,
        custom_classes_dir="custom_classes"
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.custom_classes_dir = Path(custom_classes_dir)

        self.model = YOLO(self.model_path)

        self.custom_classes_dir.mkdir(exist_ok=True)

        self.custom_classes = {}
        self.reload_custom_classes()

    def reload_custom_classes(self):
        self.custom_classes = {}

        for class_dir in self.custom_classes_dir.iterdir():
            if class_dir.is_dir():
                images = [
                    p for p in class_dir.iterdir()
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]

                if images:
                    self.custom_classes[class_dir.name] = images

        print(
            f"Loaded {len(self.custom_classes)} custom classes: "
            f"{', '.join(self.custom_classes.keys()) or 'none'}"
        )

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        result = results[0]

        annotated_frame = result.plot()

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label = self.model.names[class_id]

                detections.append(
                    {
                        "label": label,
                        "confidence": confidence
                    }
                )

        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return annotated_frame, detections
