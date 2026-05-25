#!/usr/bin/env bash
# Idempotent Azure ML setup: workspace → compute → environment → job submit
# Safe to re-run; skips already-created resources.
# Run from the project root: bash azure/provision.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"
RESOURCE_GROUP="arabic-ocr-rg"
LOCATION="eastus"
WORKSPACE="arabic-ocr-ws"
COMPUTE_NAME="cpu-cluster"
COMPUTE_SIZE="Standard_DS4_v2"   # 8 vCPUs, 28 GB RAM — CPU training (~$0.46/hr)
MIN_NODES=0                       # scales to 0 when idle → no cost at rest
MAX_NODES=1

# ── 0. Prerequisites ──────────────────────────────────────────────────────────
if ! command -v az &>/dev/null; then
  echo "ERROR: Azure CLI not found. Install with: brew install azure-cli"
  exit 1
fi
az extension add -n ml --only-show-errors 2>/dev/null || true

if [ -z "$SUBSCRIPTION_ID" ]; then
  SUBSCRIPTION_ID=$(az account show --query id -o tsv)
fi
echo "Using subscription: $SUBSCRIPTION_ID"
az account set --subscription "$SUBSCRIPTION_ID"

# ── 1. Resource Group (idempotent) ────────────────────────────────────────────
echo ""
echo "==> Resource group: $RESOURCE_GROUP"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
echo "    OK"

# ── 2. Azure ML Workspace (idempotent) ────────────────────────────────────────
echo "==> AzureML workspace: $WORKSPACE"
if az ml workspace show --name "$WORKSPACE" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  echo "    Already exists — skipping"
else
  echo "    Creating (takes ~2 min)..."
  az ml workspace create \
    --name "$WORKSPACE" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
  echo "    Created"
fi

# ── 3. CPU Compute Cluster (idempotent) ───────────────────────────────────────
echo "==> Compute cluster: $COMPUTE_NAME ($COMPUTE_SIZE)"
if az ml compute show --name "$COMPUTE_NAME" --resource-group "$RESOURCE_GROUP" \
     --workspace-name "$WORKSPACE" &>/dev/null; then
  echo "    Already exists — skipping"
else
  echo "    Creating..."
  az ml compute create \
    --name "$COMPUTE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    --type amlcompute \
    --size "$COMPUTE_SIZE" \
    --min-instances "$MIN_NODES" \
    --max-instances "$MAX_NODES" \
    --output none
  echo "    Created. Scales to 0 when idle (no cost at rest)."
fi

# ── 4. Register Environment ───────────────────────────────────────────────────
echo "==> Registering Azure ML environment (builds Docker image ~5-10 min first time)"
az ml environment create \
  --file azure/environment.yaml \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --output none
echo "    OK"

# ── 5. Submit Training Job ────────────────────────────────────────────────────
echo ""
echo "==> Submitting training job..."
JOB_NAME=$(az ml job create \
  --file azure/job.yaml \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --query name -o tsv)

echo ""
echo "========================================================"
echo " Job submitted: $JOB_NAME"
echo "========================================================"
echo ""
echo " Track in Azure ML Studio:"
echo "   https://ml.azure.com"
echo "   Workspace: $WORKSPACE  |  Job: $JOB_NAME"
echo ""
echo " Stream logs live:"
echo "   az ml job stream --name $JOB_NAME \\"
echo "     --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE"
echo ""
echo " After training finishes, run:"
echo "   bash azure/download-and-serve.sh $JOB_NAME"
echo ""
echo " This downloads best.pt and starts the FastAPI server on http://localhost:8000"
