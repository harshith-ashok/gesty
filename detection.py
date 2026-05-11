import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pips[0]].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, Middle, Ring, Pinky
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

    # Pinch controls slider
    pinch_dist = distance(thumb_tip, index_tip)
    if pinch_dist < 0.05:
        # Convert vertical index finger position to 100% -> 0%
        # Top of screen = 100%, bottom = 0%
        value = int((1.0 - index_tip.y) * 100)
        value = max(0, min(100, value))
        return "PINCH", value

    # Pointing gestures: only index finger extended
    if fingers == [0, 1, 0, 0, 0]:
        # Compare tip and PIP position to determine direction
        if index_tip.y < index_pip.y:
            return "POINT UP", None
        elif index_tip.y > index_pip.y:
            return "POINT DOWN", None
        else:
            return "POINTING", None

    # Existing gestures
    total = sum(fingers)

    if total == 0:
        return "FIST", None
    elif total == 5:
        return "OPEN PALM", None
    elif fingers == [0, 1, 1, 0, 0]:
        return "PEACE", None
    elif fingers == [1, 0, 0, 0, 0]:
        # Rough thumbs up check
        if thumb_tip.y < thumb_ip.y:
            return "THUMBS UP", None

    return "UNKNOWN", None


def draw_slider(frame, value):
    x = 50
    y = 100
    w = 40
    h = 300

    # Outline
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Fill from bottom to top
    fill_h = int(h * value / 100)
    cv2.rectangle(
        frame,
        (x, y + h - fill_h),
        (x + w, y + h),
        (0, 255, 0),
        -1
    )

    # Percentage text
    cv2.putText(
        frame,
        f"{value}%",
        (x - 10, y + h + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

slider_value = 50

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "NO HAND"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                gesture, value = detect_gesture(hand_landmarks)

                if gesture == "PINCH" and value is not None:
                    slider_value = value

        # Display gesture
        cv2.putText(
            frame,
            gesture,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        draw_slider(frame, slider_value)

        cv2.imshow("Hand Gesture Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
