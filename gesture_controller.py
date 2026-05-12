import cv2
import mediapipe as mp
import math
import time
import json
import requests
from pathlib import Path

from object_classifier import ObjectClassifier

STATE_FILE = Path(__file__).with_name("device_state.json")
CONTROL_SERVER_URL = "http://localhost:8120"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)

    default_state = {
        "devices": {
            "Main Light": {
                "status": 0
            },
            "Ceiling Fan": {
                "status": 0,
                "value": 0
            },
            "Accent Light": {
                "status": 0,
                "color": "red"
            }
        },
        "detected_objects": [],
        "last_updated": None
    }

    save_state(default_state)
    return default_state


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def sync_detected_objects(detections):
    labels = []
    seen = set()

    for detection in detections:
        label = str(detection.get("label", "")).strip().lower()

        if not label:
            continue

        if label == "person":
            continue

        if label in seen:
            continue

        seen.add(label)
        labels.append(label)

    labels.sort()

    if not labels:
        return

    try:
        requests.post(
            f"{CONTROL_SERVER_URL}/objects",
            json={"objects": labels},
            timeout=2
        )
    except Exception:
        pass


def update_device(device, status, color=None, value=None):
    state = load_state()

    state["devices"][device]["status"] = status

    if color is not None:
        state["devices"][device]["color"] = color

    if value is not None:
        state["devices"][device]["value"] = value

    save_state(state)
    print(json.dumps(state, indent=2))


def get_device_state(device):
    state = load_state()
    return state["devices"][device]


def toggle_device(device):
    current = get_device_state(device)
    current_status = current.get("status", 0)

    new_status = 0 if current_status == 1 else 1

    if device == "Ceiling Fan":
        if new_status == 1:
            value = current.get("value", 50)
            if value == 0:
                value = 50
        else:
            value = 0

        update_device(device, new_status, value=value)
    else:
        update_device(device, new_status)


def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    fingers = []

    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pips[0]].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def detect_gesture(hand_landmarks):
    fingers = fingers_up(hand_landmarks)

    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]

    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    pinch_dist = distance(thumb_tip, index_tip)

    if pinch_dist < 0.05:
        value = int((1.0 - index_tip.y) * 100)
        value = max(0, min(100, value))
        return "PINCH", value

    if fingers == [0, 1, 0, 0, 0]:
        if index_tip.y < index_pip.y:
            return "POINT UP", None
        elif index_tip.y > index_pip.y:
            return "POINT DOWN", None

    if sum(fingers) == 0:
        return "FIST", None

    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE", None

    if fingers == [1, 0, 0, 0, 0]:
        if thumb_tip.y < thumb_ip.y:
            return "THUMBS UP", None

    return "UNKNOWN", None


def draw_slider(frame, value):
    x = 50
    y = 100
    w = 40
    h = 300

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    fill_h = int(h * value / 100)

    cv2.rectangle(
        frame,
        (x, y + h - fill_h),
        (x + w, y + h),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        f"{value}%",
        (x - 10, y + h + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


def main():
    classifier = ObjectClassifier()
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    slider_value = load_state()["devices"]["Ceiling Fan"].get("value", 0)

    current_gesture = None
    gesture_start_time = 0
    gesture_sent = False
    last_pinch_update = 0
    last_object_sync = 0
    object_sync_interval = 1.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            annotated_frame, detections = classifier.detect(frame)

            now = time.time()
            if now - last_object_sync >= object_sync_interval:
                sync_detected_objects(detections)
                last_object_sync = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            detected_gesture = "NO HAND"
            value = None
            held_time = 0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    detected_gesture, value = detect_gesture(hand_landmarks)

                    if detected_gesture != current_gesture:
                        current_gesture = detected_gesture
                        gesture_start_time = time.time()
                        gesture_sent = False
                        last_pinch_update = 0

                    held_time = time.time() - gesture_start_time

                    if detected_gesture == "PINCH" and value is not None:
                        slider_value = value

                        if held_time >= 3:
                            if now - last_pinch_update >= 1:
                                update_device(
                                    "Ceiling Fan",
                                    1 if slider_value > 0 else 0,
                                    value=slider_value
                                )
                                last_pinch_update = now

                    elif held_time >= 3 and not gesture_sent:
                        if detected_gesture == "POINT UP":
                            toggle_device("Main Light")

                        elif detected_gesture == "POINT DOWN":
                            update_device("Main Light", 0)

                        elif detected_gesture == "PEACE":
                            toggle_device("Accent Light")

                        elif detected_gesture == "THUMBS UP":
                            current = get_device_state("Accent Light")

                            if current.get("status", 0) == 1:
                                update_device("Accent Light", 0)
                            else:
                                update_device(
                                    "Accent Light",
                                    1,
                                    color="red"
                                )

                        elif detected_gesture == "FIST":
                            update_device("Main Light", 0)
                            update_device("Ceiling Fan", 0, value=0)
                            update_device("Accent Light", 0)
                            slider_value = 0

                        gesture_sent = True
            else:
                current_gesture = None
                gesture_start_time = 0
                gesture_sent = False
                last_pinch_update = 0

            cv2.putText(
                annotated_frame,
                detected_gesture,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                annotated_frame,
                f"HOLD: {int(held_time)}s / 3s",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            if detections:
                visible_objects = []

                for detection in detections:
                    label = str(detection.get("label", "")).strip().lower()

                    if label and label != "person" and label not in visible_objects:
                        visible_objects.append(label)

                if visible_objects:
                    cv2.putText(
                        annotated_frame,
                        f"OBJECTS: {', '.join(visible_objects[:3])}",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2
                    )

            draw_slider(annotated_frame, slider_value)

            cv2.imshow(
                "Hand Gesture Smart Home Controller",
                annotated_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                classifier.reload_custom_classes()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
