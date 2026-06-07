# ── Stage: runtime ────────────────────────────────────────────────────────────
# python:3.11-slim = small Debian image with Python — no desktop/GUI libraries.
FROM python:3.11-slim

# System libraries that OpenCV needs on Linux (it uses libGL for image I/O).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# All app files live here inside the container.
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements-server.txt .

# Step 1 — CPU-only PyTorch BEFORE ultralytics.
# Without this, ultralytics pulls the full CUDA build (~2 GB of NVIDIA libs).
# The CPU wheel is ~200 MB — much faster and we don't have a GPU anyway.
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2 — everything else (ultralytics will reuse the torch already installed).
# grep -v filters out comment lines and the bare "pip install" note in the file.
RUN pip install --no-cache-dir \
        $(grep -v '^\s*#\|^\s*pip install\|^\s*$' requirements-server.txt | tr '\n' ' ')

# Step 3 — PaddlePaddle CPU (separate index hosted by Baidu).
RUN pip install --no-cache-dir paddlepaddle==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Step 4 — Downgrade NumPy to 1.x inside the container only.
# PaddlePaddle and several other ML libs are compiled against NumPy 1.x
# and will crash on NumPy 2.x. The local dev env is unaffected.
RUN pip install --no-cache-dir "numpy<2.0"

# ── Pre-download PaddleOCR Arabic model ──────────────────────────────────────
# PaddleOCR downloads ~50 MB of weights on first use.
# Doing it here bakes the weights into the image so the container starts
# instantly in production without needing internet access.
RUN python -c "\
from paddleocr import TextRecognition; \
TextRecognition(model_name='arabic_PP-OCRv5_mobile_rec')" || true

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/      ./src/
COPY configs/  ./configs/

# ── Copy YOLO weights ─────────────────────────────────────────────────────────
# The weights file must exist locally before running `docker build`.
# Download it first:  (already done in this session via GitHub Releases)
COPY runs/train/arabic_id_detector/weights/best.pt \
     ./runs/train/arabic_id_detector/weights/best.pt

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# Uvicorn serves the FastAPI app.
# --workers 1  → one process (PaddleOCR is not fork-safe; use K8s replicas instead)
# --host 0.0.0.0 → listen on all interfaces inside the container
CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
