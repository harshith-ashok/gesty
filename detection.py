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

    # Other four fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def detect_gesture(hand_landmarks):
    fingers = fingers_up(hand_landmarks)
    total = sum(fingers)

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # Pinch detection
    if distance(thumb_tip, index_tip) < 0.05:
        return "PINCH"

    # Basic gestures
    if total == 0:
        return "FIST"
    elif total == 5:
        return "OPEN PALM"
    elif fingers == [0, 1, 0, 0, 0]:
        return "POINTING"
    elif fingers == [0, 1, 1, 0, 0]:
        return "PEACE"
    elif fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP"

    return "UNKNOWN"


cap = cv2.VideoCapture(0)

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
                gesture = detect_gesture(hand_landmarks)

        cv2.putText(
            frame,
            gesture,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow("Hand Gesture Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
