"""
Lightweight Detection Server - YOLO Only
Runs object detection without MediaPipe gesture detection.
Lowest resource consumption, best for pure object detection use case.

Usage:
    python detect_only.py
    
This script:
- Runs YOLO object detection
- Updates device_state.json with detected objects
- Does NOT detect hand gestures
- Does NOT provide visualization (unless GESTY_DISABLE_VISUALIZATION=false)

Environment Variables:
    GESTY_SKIP_FRAMES: Skip every N frames (default 1)
    GESTY_DISABLE_VISUALIZATION: Hide camera window (default true for this script)
    GESTY_LOWER_RESOLUTION: Use 640x480 (default false)
"""

import cv2
import time
import os
import json
from pathlib import Path
from object_classifier import ObjectClassifier
from camera_utils import open_camera, close_camera, is_jetson_nano

STATE_FILE = Path(__file__).with_name("device_state.json")
SKIP_FRAMES = int(os.getenv("GESTY_SKIP_FRAMES", "1"))
DISABLE_VISUALIZATION = os.getenv(
    "GESTY_DISABLE_VISUALIZATION", "true").lower() == "true"
LOWER_RESOLUTION = os.getenv(
    "GESTY_LOWER_RESOLUTION", "false").lower() == "true"
CAMERA_USE_GSTREAMER = os.getenv(
    "GESTY_USE_GSTREAMER", "true").lower() == "true"
CAMERA_INDEX = int(os.getenv("GESTY_CAMERA_INDEX", "0"))


def sync_detected_objects(detections):
    """Update device_state.json with detected objects"""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {"detected_objects": [], "devices": {}}

    labels = []
    seen = set()

    for detection in detections:
        label = str(detection.get("label", "")).strip().lower()
        if label and label != "person" and label not in seen:
            seen.add(label)
            labels.append(label)

    state["detected_objects"] = sorted(labels)

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    return labels


def main():
    print(f"Starting lightweight YOLO detection (skip_frames={SKIP_FRAMES})")
    print(
        f"Visualization: {'disabled' if DISABLE_VISUALIZATION else 'enabled'}")
    print(f"Jetson Nano detected: {is_jetson_nano()}")
    print(f"GStreamer enabled: {CAMERA_USE_GSTREAMER}")

    classifier = ObjectClassifier()

    # Open camera with GStreamer support for RPi camera
    cap = open_camera(
        camera_index=CAMERA_INDEX,
        use_gstreamer=CAMERA_USE_GSTREAMER and is_jetson_nano(),
        lower_resolution=LOWER_RESOLUTION
    )

    if cap is None:
        print("ERROR: Could not open camera. Check /dev/video0 and GStreamer installation.")
        return

    frame_count = 0
    last_object_sync = 0
    object_sync_interval = 1.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Skip frames for resource optimization
            if frame_count % SKIP_FRAMES != 0:
                if not DISABLE_VISUALIZATION:
                    cv2.imshow("YOLO Detection Only", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # Run detection
            annotated_frame, detections = classifier.detect(frame)

            # Update state periodically
            now = time.time()
            if now - last_object_sync >= object_sync_interval:
                labels = sync_detected_objects(detections)
                print(f"[{now:.1f}] Detected: {labels}")
                last_object_sync = now

            if not DISABLE_VISUALIZATION:
                cv2.putText(
                    annotated_frame,
                    f"Objects: {len(detections)}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("YOLO Detection Only", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    classifier.reload_custom_classes()

    except KeyboardInterrupt:
        print("\nShutdown requested")
    finally:
        close_camera(cap)
        cv2.destroyAllWindows()
        print("Detection server stopped")


if __name__ == "__main__":
    main()
