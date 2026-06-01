#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-command setup for Arabic ID OCR on a fresh clone
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
WEIGHTS_DIR="runs/train/arabic_id_detector/weights"
WEIGHTS_PATH="$WEIGHTS_DIR/best.pt"

# ── 1. Python virtual environment ────────────────────────────────────────────
echo "▶ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ── 2. Core dependencies ──────────────────────────────────────────────────────
echo "▶ Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements-server.txt -q

# ── 3. PaddlePaddle (CPU) ─────────────────────────────────────────────────────
echo "▶ Installing PaddlePaddle (CPU)..."
pip install paddlepaddle==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ -q

# ── 4. YOLO weights ───────────────────────────────────────────────────────────
mkdir -p "$WEIGHTS_DIR"

if [ -f "$WEIGHTS_PATH" ]; then
    echo "▶ YOLO weights already present — skipping download."
else
    echo "▶ Downloading YOLO weights..."

    # ── Option A: GitHub Releases (recommended) ───────────────────────────────
    # After uploading best.pt to GitHub Releases, update this URL:
    RELEASE_URL="https://github.com/OmarElGohary/arabic-id-ocr/releases/latest/download/best.pt"

    if curl --output /dev/null --silent --head --fail "$RELEASE_URL"; then
        curl -L "$RELEASE_URL" -o "$WEIGHTS_PATH"
        echo "   Downloaded from GitHub Releases ✓"
    else
        # ── Option B: Manual copy fallback ────────────────────────────────────
        echo ""
        echo "   ⚠️  Could not download from GitHub Releases."
        echo "   Copy best.pt manually:"
        echo "   scp user@your-mac:path/to/best.pt $WEIGHTS_PATH"
        echo ""
    fi
fi

# ── 5. PaddleOCR model pre-download ───────────────────────────────────────────
echo "▶ Pre-downloading PaddleOCR Arabic model (arabic_PP-OCRv5_mobile_rec)..."
python3 - <<'EOF'
try:
    from paddleocr import TextRecognition
    rec = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")
    print("   PaddleOCR Arabic model ready ✓")
except Exception as e:
    print(f"   Warning: {e}")
EOF

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Run camera test:"
echo "  source venv/bin/activate"
echo "  python3 src/serving/test_inference.py --camera"
echo ""
echo "  Or start FastAPI server:"
echo "  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000"
echo "════════════════════════════════════════"
