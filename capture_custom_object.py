from pathlib import Path
import cv2
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python capture_custom_object.py <object_name>")
        return

    object_name = sys.argv[1]
    output_dir = Path("custom_objects") / object_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    count = len(list(output_dir.glob("*.jpg")))

    print(f"Capturing images for '{object_name}'")
    print("Press SPACE to save image")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.putText(
            frame,
            f"{object_name} - Images: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Capture Custom Object", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            filename = output_dir / f"{count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Saved {filename}")
            count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
