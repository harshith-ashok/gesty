#!/bin/bash
# Gesty Startup Script for Jetson Nano
# Usage: ./startup.sh [mode]
#
# Modes:
#   full       - Full features with visualization
#   lite       - Lite mode (recommended for 4GB Nano)
#   api        - API server only (no camera)
#   detect     - YOLO detection only (no gestures)
#   headless   - Full features but no visualization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Installing dependencies..."
    pip install -q -r "$SCRIPT_DIR/requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

MODE="${1:-lite}"

echo "=== Gesty Jetson Nano Startup ==="
echo "Mode: $MODE"
echo "Python: $(which python3)"
echo ""

case "$MODE" in
    full)
        echo "Starting full mode with visualization..."
        cd "$SCRIPT_DIR"
        python gesture_controller.py
        ;;
    lite)
        echo "Starting lite mode (recommended)..."
        export GESTY_LITE_MODE=true
        export GESTY_SKIP_FRAMES=2
        export GESTY_LOWER_RESOLUTION=true
        cd "$SCRIPT_DIR"
        python gesture_controller.py
        ;;
    api)
        echo "Starting API server only..."
        export GESTY_API_HOST=0.0.0.0
        export GESTY_API_PORT=8120
        cd "$SCRIPT_DIR"
        python api_server.py
        ;;
    detect)
        echo "Starting YOLO detection only (minimal resources)..."
        export GESTY_SKIP_FRAMES=2
        export GESTY_DISABLE_VISUALIZATION=true
        cd "$SCRIPT_DIR"
        python detect_only.py
        ;;
    headless)
        echo "Starting full features without visualization..."
        export GESTY_DISABLE_VISUALIZATION=true
        export GESTY_SKIP_FRAMES=2
        cd "$SCRIPT_DIR"
        python gesture_controller.py
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  full       - Full features with visualization"
        echo "  lite       - Lite mode (recommended for 4GB Nano)"
        echo "  api        - API server only (no camera)"
        echo "  detect     - YOLO detection only (no gestures)"
        echo "  headless   - Full features but no visualization"
        exit 1
        ;;
esac
