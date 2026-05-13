"""
Camera utilities for Jetson Nano with Raspberry Pi Camera support via GStreamer
Compatible with Raspberry Pi Camera Module v2 connected to the CSI port.
"""

import cv2
from typing import Optional


class CameraConfig:
    """Predefined camera configurations"""

    # Default CSI camera pipeline for Jetson Nano
    JETSON_NANO_CSI = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=1280, height=720, "
        "format=NV12, "
        "framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, "
        "width=1280, height=720, "
        "format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )

    # Lower resolution mode for better performance
    JETSON_NANO_CSI_LOW = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=640, height=480, "
        "format=NV12, "
        "framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, "
        "width=640, height=480, "
        "format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )

    # USB webcam fallback
    USB_CAMERA = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=1280, height=720, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )


def is_jetson_nano() -> bool:
    """Check whether this system is a Jetson Nano"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "Jetson Nano" in f.read()
    except Exception:
        return False


def get_gstreamer_pipeline(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0
) -> str:
    """
    Create a GStreamer pipeline for the CSI camera.
    """

    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), "
        f"width={width}, height={height}, "
        f"format=NV12, "
        f"framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, "
        f"width={width}, height={height}, "
        f"format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )


def open_camera(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    use_gstreamer: bool = True,
    lower_resolution: bool = False,
    flip_method: int = 0
) -> Optional[cv2.VideoCapture]:
    """
    Open the camera using GStreamer (preferred) or V4L2 fallback.
    """

    if lower_resolution:
        width, height = 640, 480

    cap = None

    # Try CSI camera via GStreamer
    if use_gstreamer:
        pipeline = get_gstreamer_pipeline(
            width=width,
            height=height,
            fps=fps,
            flip_method=flip_method
        )

        print("\nUsing GStreamer pipeline:")
        print(pipeline)
        print()

        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if cap.isOpened():
            print(
                f"✓ Opened CSI camera via GStreamer: {width}x{height}@{fps} FPS")
            return cap
        else:
            print("✗ Failed to open CSI camera via GStreamer")

    # Fallback to USB/V4L2 camera
    print("Trying V4L2 fallback...")

    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(
            f"✓ Opened camera via V4L2: "
            f"{actual_width}x{actual_height}@{actual_fps:.1f} FPS"
        )
        return cap

    print("✗ Failed to open any camera")
    return None


def close_camera(cap: cv2.VideoCapture) -> None:
    """Release the camera"""
    if cap is not None:
        cap.release()
        print("Camera closed")


if __name__ == "__main__":
    import time

    print(f"Jetson Nano detected: {is_jetson_nano()}")

    cap = open_camera(
        use_gstreamer=True,
        lower_resolution=True,   # Start with 640x480 for reliability
        fps=30
    )

    if cap is None:
        print("Camera initialization failed.")
        exit(1)

    print("Testing camera for 5 seconds...")

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_count += 1

    elapsed = time.time() - start_time

    if frame_count > 0:
        measured_fps = frame_count / elapsed
        print(
            f"Captured {frame_count} frames in "
            f"{elapsed:.2f} seconds "
            f"({measured_fps:.2f} FPS)"
        )
    else:
        print("No frames were captured.")

    close_camera(cap)
