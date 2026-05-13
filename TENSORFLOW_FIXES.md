# Jetson Nano - TensorFlow & h5py Troubleshooting

## The Problem

TensorFlow 2.14.0+ and h5py have compatibility issues on ARM64 architectures (Jetson Nano). Build failures typically show:

```
error: invalid command 'bdist_wheel'
error: command 'gcc' failed with exit status 1
```

## Solutions

### Solution 1: Use Official NVIDIA PyTorch (Recommended)

Skip TensorFlow entirely and use PyTorch which has better Jetson support.

```bash
pip install torch torchvision torchaudio
```

### Solution 2: Use Pre-built JetPack Image

NVIDIA provides optimized JetPack images with TensorFlow pre-installed and h5py fixed.

```bash
# JetPack 4.6+ includes working h5py
# Flash JetPack from: https://developer.nvidia.com/jetson-nano-sd-card-image
```

### Solution 3: Install via conda-forge (if using conda)

```bash
conda install -c conda-forge tensorflow h5py
```

### Solution 4: Force reinstall h5py

```bash
pip uninstall -y h5py
pip install --no-binary h5py h5py==3.1.0
```

### Solution 5: Downgrade TensorFlow

If you must use TensorFlow:

```bash
pip install tensorflow==2.13.1 h5py==2.10.0
```

## Gesty-Specific: No TensorFlow Needed!

**Important**: Gesty's main codebase (gesture detection + object detection) does NOT require TensorFlow.

- **Gesture Detection**: Uses MediaPipe (Lite models, highly optimized)
- **Object Detection**: Uses YOLO (direct inference, no TensorFlow)
- **Training**: Only `train_custom_model.py` uses TensorFlow (optional, can run on desktop)

### For Jetson Nano Deployment

Use only `requirements.txt` (no TensorFlow):

```bash
pip install -r requirements.txt
# TensorFlow not included!
```

### For Model Training

Train on desktop/laptop with `dev-requirements.txt`, then deploy `.pt` files to Jetson:

```bash
# On desktop:
pip install -r dev-requirements.txt
python train_custom_model.py

# Copy trained model to Jetson
scp custom_model/model.keras jetson@192.168.1.100:~/gesty/
```

## If You Really Need TensorFlow on Jetson

### Option A: Use TensorFlow Lite

```bash
pip install tflite-runtime
```

Convert your models:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('model_dir')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Option B: Use NVIDIA NVIDIA L4T Container

```bash
docker run --runtime nvidia --rm -it nvcr.io/nvidia/l4t-tensorflow:latest
# Pre-configured TensorFlow environment
```

## Verification

Check what's installed:

```bash
# MediaPipe (✓ needed)
python -c "import mediapipe; print(mediapipe.__version__)"

# YOLO (✓ needed)
python -c "from ultralytics import YOLO; print('YOLO OK')"

# TensorFlow (✗ not needed for inference)
python -c "import tensorflow as tf" 2>&1 || echo "TensorFlow not installed (OK for Jetson runtime)"

# h5py (only if using TensorFlow)
python -c "import h5py; print(h5py.__version__)" 2>&1 || echo "h5py not installed (OK)"
```

## Environment Check Script

```bash
#!/bin/bash
echo "=== Gesty Jetson Environment Check ==="
python3 -c "
import platform
import sys
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Arch: {platform.machine()}')

pkgs = ['mediapipe', 'ultralytics', 'opencv', 'numpy', 'pydantic', 'fastapi']
for pkg in pkgs:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg}')
"
```

## Quick Fix Checklist

- [ ] Using official JetPack OS (includes optimized libraries)
- [ ] TensorFlow NOT in requirements.txt
- [ ] Running lite mode: `export GESTY_LITE_MODE=true`
- [ ] Skip frames enabled: `export GESTY_SKIP_FRAMES=2`
- [ ] Visualization disabled: `export GESTY_DISABLE_VISUALIZATION=true`
- [ ] Using AP I-only mode if gesture control not needed
