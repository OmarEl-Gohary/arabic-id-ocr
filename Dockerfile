FROM python:3.11-slim

# System libraries required by OpenCV on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server.txt .

# Step 1 — CPU-only PyTorch before ultralytics to avoid the 2 GB CUDA build
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2 — all other dependencies
RUN pip install --no-cache-dir -r requirements-server.txt

# Step 3 — PaddlePaddle CPU (hosted on Baidu's index, requires AVX2 on the host)
RUN pip install --no-cache-dir paddlepaddle==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Step 4 — Pre-download PaddleOCR Arabic model into the image so the container
# starts instantly without internet access at runtime
RUN python -c "\
from paddleocr import TextRecognition; \
TextRecognition(model_name='arabic_PP-OCRv5_mobile_rec')" || true

COPY src/      ./src/
COPY configs/  ./configs/
COPY runs/train/arabic_id_detector/weights/best.pt \
     ./runs/train/arabic_id_detector/weights/best.pt

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
