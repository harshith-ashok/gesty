# Camera Utilities Module

The `camera_utils.py` module provides a unified interface for camera access on Jetson Nano, supporting both:

- **Raspberry Pi CSI Camera** (via GStreamer `nvarguscamerasrc`)
- **USB Cameras** (fallback to V4L2)

## Usage

### Basic Camera Capture

```python
from camera_utils import open_camera, close_camera

# Open with GStreamer (recommended for RPi camera)
cap = open_camera(
    camera_index=0,           # /dev/video0
    use_gstreamer=True,       # Use GStreamer backend
    width=1280,
    height=720,
    fps=30
)

if cap is None:
    print("Failed to open camera")
    exit(1)

# Capture frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    # ...

close_camera(cap)
```

### Detect Jetson Nano Hardware

```python
from camera_utils import is_jetson_nano

if is_jetson_nano():
    print("Running on Jetson Nano - GStreamer available")
    use_gstreamer = True
else:
    print("Not Jetson Nano - fallback to V4L2")
    use_gstreamer = False
```

### Test Camera

```bash
# Direct test
python camera_utils.py

# Output example:
# ✓ Opened camera via GStreamer: 1280x720@30fps
# Captured 150 frames in 5 seconds (30.0 FPS)
```

## How It Works

### GStreamer Pipeline (RPi Camera)

```
nvarguscamerasrc (Jetson camera source)
  ↓
video/x-raw(memory:NVMM) → NV12 format on GPU
  ↓
nvvidconv (GPU conversion)
  ↓
video/x-raw, BGRx
  ↓
videoconvert (GPU to CPU transfer)
  ↓
video/x-raw, BGR (OpenCV compatible)
  ↓
appsink (feed to Python)
```

**Benefits:**

- Uses GPU for format conversion (fast)
- Optimized for Jetson Nano hardware
- Native support for CSI camera
- Low CPU overhead

### V4L2 Fallback (USB Camera)

```
v4l2src (/dev/video0 or /dev/video1)
  ↓
OpenCV VideoCapture
  ↓
Set resolution/FPS
```

**Used when:**

- GStreamer not available
- Running on non-Jetson systems
- USB camera connected

## Performance Characteristics

### Jetson Nano with RPi CSI Camera (GStreamer)

```
Resolution: 1280x720
FPS: 30
CPU: ~5-10%
GPU: ~20-30%
Memory: ~50MB
```

### Jetson Nano with USB Camera (V4L2)

```
Resolution: 1280x720
FPS: 20-25
CPU: ~15-20%
GPU: ~5-10%
Memory: ~30MB
```

## Troubleshooting

### "GStreamer pipeline failed to open"

1. Check `nvarguscamerasrc` availability:

   ```bash
   gst-inspect-1.0 nvarguscamerasrc
   ```

2. Update NVIDIA drivers:

   ```bash
   sudo apt-get update
   sudo apt-get dist-upgrade
   ```

3. Install missing GStreamer plugins:
   ```bash
   sudo apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-good
   ```

### "Camera closed immediately after opening"

- Check CSI connector is seated properly
- Verify camera works with:
  ```bash
  nvgstcapture-1.0
  ```

### "Failed to read frame from camera"

- Check camera is not in use by another process:
  ```bash
  lsof /dev/video0
  ```
- Try fallback mode:
  ```bash
  export GESTY_USE_GSTREAMER=false
  ```

## API Reference

### `open_camera(camera_index=0, width=1280, height=720, fps=30, use_gstreamer=True, lower_resolution=False)`

Open camera with optimized settings.

**Parameters:**

- `camera_index` (int): Camera device (0=/dev/video0, 1=/dev/video1, etc)
- `width` (int): Frame width
- `height` (int): Frame height
- `fps` (int): Target frames per second
- `use_gstreamer` (bool): Use GStreamer backend
- `lower_resolution` (bool): Force 640x480

**Returns:** `cv2.VideoCapture` or `None`

### `close_camera(cap)`

Close camera and cleanup resources.

**Parameters:**

- `cap`: cv2.VideoCapture object

### `is_jetson_nano()`

Check if running on Jetson Nano hardware.

**Returns:** `bool`

### `get_gstreamer_pipeline(width=1280, height=720, fps=30)`

Generate GStreamer pipeline string.

**Returns:** Pipeline string for `cv2.VideoCapture(..., cv2.CAP_GSTREAMER)`
