# Jetson Nano Deployment Guide

## Quick Start

### 1. Prerequisites

- Jetson Nano 4GB with JetPack 4.6+ (includes optimized libraries)
- Python 3.7+
- Jetson Nano Camera or USB camera

### 2. Installation

```bash
# Clone/setup project
cd ~/gesty

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU/GPU optimized versions)
pip install -r requirements.txt
```

### 3. Running Modes

#### Standard Mode (Full Features)

```bash
source .venv/bin/activate
python gesture_controller.py &
uvicorn api:app --host 0.0.0.0 --port 8120
```

#### Lite Mode (Minimal Resource Usage)

```bash
export GESTY_LITE_MODE=true
export GESTY_SKIP_FRAMES=2
export GESTY_DISABLE_VISUALIZATION=true
python gesture_controller.py
```

#### API-Only Mode (No Camera/Visualization)

```bash
# Just run the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8120
```

#### Dashboard (Optional, Resource Heavy)

```bash
# Install optional Gradio dependency first
pip install gradio==4.14.0
python dashboard.py
```

## Camera Setup

### Raspberry Pi Camera (Recommended)

Gesty uses **GStreamer** to interface with the Raspberry Pi camera connected via CSI/MIPI:

```bash
# Test camera directly
python camera_utils.py

# Or in Python:
from camera_utils import open_camera, close_camera
cap = open_camera(use_gstreamer=True)
# ... capture frames ...
close_camera(cap)
```

**Environment Variables:**

- `GESTY_USE_GSTREAMER=true` - Enable GStreamer (default)
- `GESTY_CAMERA_INDEX=0` - Camera device (/dev/video0)

**Troubleshooting:**

- Camera not found: Check `/dev/video0` exists (`ls /dev/video*`)
- GStreamer error: Ensure `gstreamer1.0-tools` is installed (`sudo apt install gstreamer1.0-tools`)
- Fallback to V4L2: Set `GESTY_USE_GSTREAMER=false`

### USB Camera (Fallback)

For USB cameras, GStreamer will fall back to V4L2:

```bash
# Find USB camera
v4l2-ctl --list-devices

# Set camera index (usually 1 for USB if CSI is 0)
export GESTY_CAMERA_INDEX=1
python gesture_controller.py
```

## Environment Variables

| Variable                      | Default | Description                                                       |
| ----------------------------- | ------- | ----------------------------------------------------------------- |
| `GESTY_LITE_MODE`             | false   | Enable lite mode (reduced inference frequency)                    |
| `GESTY_SKIP_FRAMES`           | 1       | Skip every N frames (1=process all, 2=process every 2nd, etc)     |
| `GESTY_LOWER_RESOLUTION`      | false   | Use 640x480 instead of camera default                             |
| `GESTY_DISABLE_VISUALIZATION` | false   | Disable OpenCV display (useful for headless setup)                |
| `GESTY_USE_GSTREAMER`         | true    | Use GStreamer backend for camera (recommended for RPi CSI camera) |
| `GESTY_CAMERA_INDEX`          | 0       | Camera device index (/dev/video0, /dev/video1, etc)               |
| `GESTY_API_HOST`              | 0.0.0.0 | FastAPI server host                                               |
| `GESTY_API_PORT`              | 8120    | FastAPI server port                                               |
| `GESTY_CONFIDENCE`            | 0.5     | YOLO detection confidence (0.0-1.0)                               |

## Performance Tuning

### For Jetson Nano 4GB

**Option 1: Maximum Performance**

```bash
export GESTY_SKIP_FRAMES=1
export GESTY_LOWER_RESOLUTION=false
# ~25-30 FPS, moderate CPU/GPU usage
```

**Option 2: Balanced (Recommended)**

```bash
export GESTY_LITE_MODE=true
export GESTY_SKIP_FRAMES=2
export GESTY_LOWER_RESOLUTION=true
# ~10-15 FPS, low CPU/GPU usage
```

**Option 3: Minimum Resources (Server Only)**

```bash
export GESTY_DISABLE_VISUALIZATION=true
export GESTY_SKIP_FRAMES=3
# API-only, ~5 FPS inference
```

## Troubleshooting

### Camera Issues

**GStreamer Not Found**

```bash
# Install GStreamer tools
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-good

# Check if available
gst-launch-1.0 --version
```

**Camera Not Opening via GStreamer**

```bash
# Test with gst-launch
gst-launch-1.0 nvarguscamerasrc ! videoscale ! video/x-raw, width=1280, height=720 ! videoconvert ! ximagesink

# If error, try updating driver
sudo apt-get update && sudo apt-get upgrade nvidia-l4t*
```

**Fallback to V4L2**

```bash
# Disable GStreamer
export GESTY_USE_GSTREAMER=false
python gesture_controller.py
```

### h5py Build Errors

**Solution:** Use NVIDIA's pre-built JetPack image which includes optimized h5py.
If building manually:

```bash
pip install --upgrade --force-reinstall h5py==3.1.0
```

### Camera Not Found

- Check camera: `ls -la /dev/video*`
- For USB cameras, verify with: `v4l2-ctl --list-devices`
- Update camera index in code if needed

### Out of Memory

- Enable `GESTY_LITE_MODE=true`
- Disable visualization: `GESTY_DISABLE_VISUALIZATION=true`
- Increase skip frames: `GESTY_SKIP_FRAMES=3`
- Disable MediaPipe and use YOLO only (requires code modification)

### Slow Inference

- Use `GESTY_LOWER_RESOLUTION=true`
- Increase `GESTY_SKIP_FRAMES`
- Check GPU usage: `tegrastats`

## Systemd Service (Optional Auto-start)

Create `/etc/systemd/system/gesty.service`:

```ini
[Unit]
Description=Gesty Gesture Controller
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/gesty
Environment="GESTY_LITE_MODE=true"
Environment="GESTY_SKIP_FRAMES=2"
ExecStart=/home/jetson/gesty/.venv/bin/python gesture_controller.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable gesty.service
sudo systemctl start gesty.service
```

## Performance Benchmarks (Jetson Nano 4GB, JetPack 4.6)

| Mode          | FPS   | Memory | CPU | Notes              |
| ------------- | ----- | ------ | --- | ------------------ |
| Full          | 15-20 | 900MB  | 60% | With visualization |
| Lite (skip 2) | 8-10  | 650MB  | 40% | Reduced inference  |
| API Only      | 12-15 | 450MB  | 50% | No visualization   |

## Next Steps

1. **Raspberry Pi Camera Setup**: Modify camera capture in `gesture_controller.py`
2. **Custom Models**: Use `train_custom_model.py` on desktop, deploy `.pt` files
3. **Integration**: Extend `api.py` to control your smart home devices
