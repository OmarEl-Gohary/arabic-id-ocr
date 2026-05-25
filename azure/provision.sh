#!/usr/bin/env bash
# One-shot Azure ML setup: workspace → compute → environment → job submit
# Run from the project root: bash azure/provision.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"   # set env var or paste here
RESOURCE_GROUP="arabic-ocr-rg"
LOCATION="eastus"
WORKSPACE="arabic-ocr-ws"
COMPUTE_NAME="gpu-cluster"
COMPUTE_SIZE="Standard_NC4as_T4_v3"   # 1x NVIDIA T4, 4 vCPUs, 28 GB RAM (~$0.52/hr)
MIN_NODES=0                            # scale to 0 when idle → no cost at rest
MAX_NODES=1

# ── 0. Prerequisites ──────────────────────────────────────────────────────────
if ! command -v az &>/dev/null; then
  echo "ERROR: Azure CLI not found. Install with: brew install azure-cli"
  exit 1
fi

az extension add -n ml --only-show-errors 2>/dev/null || true

if [ -z "$SUBSCRIPTION_ID" ]; then
  echo "Fetching default subscription..."
  SUBSCRIPTION_ID=$(az account show --query id -o tsv)
fi
echo "Using subscription: $SUBSCRIPTION_ID"
az account set --subscription "$SUBSCRIPTION_ID"

# ── 1. Resource Group ─────────────────────────────────────────────────────────
echo ""
echo "==> Creating resource group: $RESOURCE_GROUP"
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

# ── 2. Azure ML Workspace ─────────────────────────────────────────────────────
echo "==> Creating AzureML workspace: $WORKSPACE (takes ~2 min)"
az ml workspace create \
  --name "$WORKSPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

# ── 3. GPU Compute Cluster ────────────────────────────────────────────────────
echo "==> Creating GPU compute cluster: $COMPUTE_NAME ($COMPUTE_SIZE)"
az ml compute create \
  --name "$COMPUTE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --type amlcompute \
  --size "$COMPUTE_SIZE" \
  --min-instances "$MIN_NODES" \
  --max-instances "$MAX_NODES" \
  --output none

echo "    Cluster created. Will auto-scale to 0 when idle (no cost at rest)."

# ── 4. Register Environment ───────────────────────────────────────────────────
echo "==> Registering Azure ML environment (builds Docker image, ~5-10 min first time)"
az ml environment create \
  --file azure/environment.yaml \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --output none

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
echo "   → Workspaces → $WORKSPACE → Jobs → $JOB_NAME"
echo ""
echo " Or stream logs here:"
echo "   az ml job stream --name $JOB_NAME \\"
echo "     --resource-group $RESOURCE_GROUP \\"
echo "     --workspace-name $WORKSPACE"
echo ""
echo " After training, download best weights:"
echo "   az ml job download --name $JOB_NAME \\"
echo "     --resource-group $RESOURCE_GROUP \\"
echo "     --workspace-name $WORKSPACE \\"
echo "     --output ./azure-outputs/"
echo ""
echo " MLflow experiments are auto-tracked in Azure ML Studio"
echo " under the 'arabic-ocr-detection' experiment."
