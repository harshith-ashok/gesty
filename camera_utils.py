"""
Camera utilities for Jetson Nano with Raspberry Pi Camera support via GStreamer
"""

import cv2
import os
from typing import Optional, Tuple


class CameraConfig:
    """Camera configuration for different backends"""

    # GStreamer pipeline for Raspberry Pi camera on Jetson Nano
    JETSON_NANO_CSI = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )

    # GStreamer pipeline for USB camera (fallback)
    USB_CAMERA = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=1280, height=720, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )

    # Simple V4L2 (fallback if GStreamer not available)
    V4L2_DEFAULT = "/dev/video0"


def get_gstreamer_pipeline(width: int = 1280, height: int = 720, fps: int = 30) -> str:
    """
    Generate GStreamer pipeline for Raspberry Pi camera on Jetson Nano

    Args:
        width: Frame width
        height: Frame height
        fps: Frames per second

    Returns:
        GStreamer pipeline string
    """
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink"
    )


def open_camera(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    use_gstreamer: bool = True,
    lower_resolution: bool = False
) -> Optional[cv2.VideoCapture]:
    """
    Open camera with optimized settings for Jetson Nano

    Args:
        camera_index: Camera device index (0 for /dev/video0)
        width: Frame width
        height: Frame height
        fps: Target frames per second
        use_gstreamer: Use GStreamer backend (recommended for RPi camera)
        lower_resolution: Use lower resolution (640x480) for resource-constrained mode

    Returns:
        cv2.VideoCapture object or None if failed
    """

    if lower_resolution:
        width, height = 640, 480

    cap = None

    # Try GStreamer first (recommended for Jetson Nano CSI camera)
    if use_gstreamer:
        try:
            pipeline = get_gstreamer_pipeline(width, height, fps)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if cap.isOpened():
                print(
                    f"✓ Opened camera via GStreamer: {width}x{height}@{fps}fps")
                return cap
            else:
                print("✗ GStreamer pipeline failed to open")
        except Exception as e:
            print(f"✗ GStreamer error: {e}")

    # Fallback to V4L2 (for USB cameras)
    try:
        cap = cv2.VideoCapture(camera_index)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            print(
                f"✓ Opened camera via V4L2: {actual_width}x{actual_height}@{actual_fps}fps")
            return cap
    except Exception as e:
        print(f"✗ V4L2 error: {e}")

    print("✗ Failed to open camera")
    return None


def close_camera(cap: cv2.VideoCapture) -> None:
    """Close camera and cleanup resources"""
    if cap is not None:
        cap.release()
        print("Camera closed")


# Check if running on Jetson Nano
def is_jetson_nano() -> bool:
    """Check if running on Jetson Nano hardware"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read()
            return "Jetson Nano" in model
    except:
        return False


if __name__ == "__main__":
    print(f"Jetson Nano detected: {is_jetson_nano()}")

    # Test camera opening
    cap = open_camera(use_gstreamer=True, lower_resolution=False)

    if cap:
        print(f"Testing camera for 5 seconds...")
        import time
        start = time.time()
        frames = 0

        while time.time() - start < 5:
            ret, frame = cap.read()
            if ret:
                frames += 1

        fps = frames / 5
        print(f"Captured {frames} frames in 5 seconds ({fps:.1f} FPS)")
        close_camera(cap)
