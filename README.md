# Gesty

simple `mediapipe` based gesture detector using python.

## Phase 1 goals

- [x] basic gesture detection
- [x] covert detected gestures to api response
- [x] integrate with [litey](https://github.com/harshith-ashok/litey)

## Current iteration

1. Gradio UI for dashboard
2. FastAPI api for litey integration
3. `device_state.json` file for rudimentary status and session storage

## Phase 2 goals

- [x] deploy in Jetson Nano (with optimization for resource constraints)
- [ ] Add rpi camera support and verify
- [ ] Native audio controls

## Jetson Nano Deployment

See [JETSON_SETUP.md](JETSON_SETUP.md) for detailed Jetson Nano deployment guide.

### Camera Setup

Gesty uses **GStreamer** to efficiently interface with Raspberry Pi CSI camera on Jetson Nano:

```bash
# Test camera with GStreamer
python camera_utils.py

# Check camera device
ls /dev/video0
```

### Quick Start on Jetson Nano

```bash
# Load lite mode config
source .env.jetson

# Run gesture controller + API
python gesture_controller.py &
python api_server.py
```

Or use the startup script:

```bash
./startup.sh lite   # Recommended for 4GB Nano
./startup.sh api    # API only
./startup.sh detect # Object detection only
```
