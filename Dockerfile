# Bouldering Analysis — FastAPI Backend
# CPU-only image for Render (and future VPS) deployment.
# pyproject.toml pins torch to the CUDA 12.8 index; this Dockerfile
# overrides that with the CPU build to keep the image size manageable.

FROM python:3.11-slim

ARG OPENCV_VERSION=4.12.0.88

# System dependencies required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app

# --- PyTorch (CPU build, installed before other deps) ---
# Installs CPU-only torch/torchvision instead of the CUDA build referenced
# in pyproject.toml. The CPU wheel is ~700 MB vs ~2.9 GB for CUDA.
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

# --- Application dependencies ---
# All versions are pinned here for reproducible builds. Keep these in sync with
# pyproject.toml when updating dependencies. Intentional deviations:
#   - torch/torchvision: CPU build from download.pytorch.org/whl/cpu (not CUDA)
#   - opencv-python-headless: replaces opencv-python (same cv2 API, no GUI libs)
#   - xgboost/scikit-learn/joblib: range pins (>=x,<y) intentionally match pyproject.toml
RUN pip install --no-cache-dir \
    "fastapi==0.128.6" \
    "uvicorn[standard]==0.40.0" \
    "pydantic-settings==2.7.1" \
    "httpx==0.28.1" \
    "python-multipart==0.0.22" \
    "python-json-logger==3.2.1" \
    "Pillow==12.1.1" \
    "opencv-python-headless==${OPENCV_VERSION}" \
    "numpy==2.2.6" \
    "PyYAML==6.0.2" \
    "supabase==2.27.2" \
    "networkx==3.4.2" \
    "ultralytics==8.3.233" \
    "xgboost>=2.0.0,<3.0.0" \
    "scikit-learn>=1.3.0,<2.0.0" \
    "joblib>=1.3.0,<2.0.0" \
    && (pip uninstall -y opencv-python || true) \
    && pip install --no-cache-dir --force-reinstall "opencv-python-headless==${OPENCV_VERSION}"

# --- Application source ---
COPY --chown=app:app src/ ./src/

USER app

# Render injects $PORT at runtime; 8000 is the local default.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request, sys; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT','8000') + '/health'); sys.exit(0)"

CMD uvicorn src.app:application --host 0.0.0.0 --port ${PORT:-8000}
