# Bouldering Analysis — FastAPI Backend
# CPU-only image for Render (and future VPS) deployment.
# pyproject.toml pins torch to the CUDA 12.8 index; this Dockerfile
# overrides that with the CPU build to keep the image size manageable.

FROM python:3.11-slim

# System dependencies required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch (CPU build, installed before other deps) ---
# Installs CPU-only torch/torchvision instead of the CUDA build referenced
# in pyproject.toml. The CPU wheel is ~700 MB vs ~2.9 GB for CUDA.
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

# --- Application dependencies ---
# opencv-python-headless replaces opencv-python: same cv2 API, no GUI libs.
RUN pip install --no-cache-dir \
    "fastapi==0.128.6" \
    "uvicorn[standard]==0.40.0" \
    "pydantic-settings==2.7.1" \
    "httpx==0.28.1" \
    "python-multipart==0.0.22" \
    "python-json-logger==3.2.1" \
    "Pillow==12.1.1" \
    "opencv-python-headless==4.12.0.88" \
    "numpy==2.2.6" \
    "PyYAML==6.0.2" \
    "supabase==2.27.2" \
    "networkx==3.4.2" \
    "ultralytics==8.3.233" \
    "xgboost>=2.0.0,<3.0.0" \
    "scikit-learn>=1.3.0,<2.0.0" \
    "joblib>=1.3.0,<2.0.0"

# --- Application source ---
COPY src/ ./src/

# Render injects $PORT at runtime; 8000 is the local default.
EXPOSE 8000
CMD uvicorn src.app:application --host 0.0.0.0 --port ${PORT:-8000}
