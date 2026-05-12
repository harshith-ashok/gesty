import cv2
from pathlib import Path
import sys

DATASET_DIR = Path("dataset")
IMAGES_TO_CAPTURE = 100


def main():
    if len(sys.argv) < 2:
        print("Usage: python capture_training_images.py <class_name>")
        return

    class_name = sys.argv[1].strip().lower().replace(" ", "_")
    output_dir = DATASET_DIR / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    print(f"Capturing images for class: {class_name}")
    print("Press SPACE to capture an image.")
    print("Press Q to quit.")

    count = len(list(output_dir.glob("*.jpg")))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            f"Class: {class_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Captured: {count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("Capture Training Images", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            filename = output_dir / f"{count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            count += 1
            print(f"Saved {filename}")

            if count >= IMAGES_TO_CAPTURE:
                print("Target number of images reached.")
                break

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
