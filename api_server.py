"""
Headless Gesty API Server
Runs only the FastAPI backend without gesture detection or visualization.
Useful for Jetson Nano deployments without displays or when gesture control isn't needed.

Usage:
    python api_server.py          # Standard mode
    GESTY_API_PORT=9000 python api_server.py  # Custom port
"""

import os
import uvicorn
from pathlib import Path

# Import the FastAPI app from api.py
from api import app

if __name__ == "__main__":
    host = os.getenv("GESTY_API_HOST", "0.0.0.0")
    port = int(os.getenv("GESTY_API_PORT", "8120"))

    print(f"Starting Gesty API Server on {host}:{port}")
    print(f"API endpoints: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
