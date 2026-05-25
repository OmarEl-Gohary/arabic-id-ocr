#!/usr/bin/env bash
# Download trained weights from Azure ML job and serve the API locally.
# Usage: bash azure/download-and-serve.sh <JOB_NAME>
# Run from the project root.
set -euo pipefail

RESOURCE_GROUP="arabic-ocr-rg"
WORKSPACE="arabic-ocr-ws"
WEIGHTS_DEST="./runs/train/arabic_id_detector/weights"

JOB_NAME="${1:-}"
if [ -z "$JOB_NAME" ]; then
  echo "Usage: bash azure/download-and-serve.sh <JOB_NAME>"
  echo ""
  echo "Find your job name with:"
  echo "  az ml job list --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --query '[].name' -o tsv"
  exit 1
fi

# ── 1. Wait for job to complete ───────────────────────────────────────────────
echo "==> Checking job status: $JOB_NAME"
STATUS=$(az ml job show \
  --name "$JOB_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --query status -o tsv)

if [ "$STATUS" != "Completed" ]; then
  echo "    Job status: $STATUS"
  echo "    Stream logs with:"
  echo "      az ml job stream --name $JOB_NAME --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE"
  exit 1
fi
echo "    Job completed successfully"

# ── 2. Download artifacts ─────────────────────────────────────────────────────
echo "==> Downloading job artifacts..."
DOWNLOAD_DIR="./azure-outputs/$JOB_NAME"
az ml job download \
  --name "$JOB_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --download-path "$DOWNLOAD_DIR" \
  --output none

# ── 3. Copy weights to the path serving.yaml expects ─────────────────────────
echo "==> Installing weights to $WEIGHTS_DEST"
mkdir -p "$WEIGHTS_DEST"

# Try both possible artifact paths
BEST_PT=$(find "$DOWNLOAD_DIR" -name "best.pt" | head -1)
if [ -z "$BEST_PT" ]; then
  echo "ERROR: best.pt not found in downloaded artifacts at $DOWNLOAD_DIR"
  find "$DOWNLOAD_DIR" -type f | head -20
  exit 1
fi

cp "$BEST_PT" "$WEIGHTS_DEST/best.pt"
echo "    Copied: $BEST_PT → $WEIGHTS_DEST/best.pt"

# ── 4. Install serving dependencies (if needed) ───────────────────────────────
echo "==> Checking serving dependencies..."
python -c "import easyocr" 2>/dev/null || {
  echo "    Installing easyocr..."
  pip install easyocr>=1.7.1 --quiet
}
python -c "import fastapi, uvicorn" 2>/dev/null || {
  echo "    Installing serving packages..."
  pip install fastapi uvicorn[standard] python-multipart --quiet
}
echo "    OK"

# ── 5. Start FastAPI server ───────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Starting Arabic ID OCR API (on-prem)"
echo " http://localhost:8000/docs"
echo "========================================================"
echo ""
python -m uvicorn src.serving.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
