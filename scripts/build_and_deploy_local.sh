#!/bin/bash
set -euo pipefail

# Build Docker image and deploy to local k3d cluster

MODEL_NAME="${1:-clip-test}"
CLUSTER_NAME="${2:-model-zoo-dev}"

echo "Building and deploying $MODEL_NAME to local k3d cluster..."

# Check if cluster exists
if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "Error: Cluster '$CLUSTER_NAME' not found. Start it first with:"
    echo "./scripts/dev_cluster.sh"
    exit 1
fi

# Build Docker image locally
echo "Building Docker image..."
docker build -t "model-zoo-$MODEL_NAME:latest" -f infra/docker/Dockerfile.model .

# Import image into k3d
echo "Importing image into k3d cluster..."
k3d image import "model-zoo-$MODEL_NAME:latest" -c "$CLUSTER_NAME"

# Deploy model (uses CPU in local mode)
echo "Deploying $MODEL_NAME model..."

# Use local weights file for CLIP
if [[ "$MODEL_NAME" == *"clip"* ]]; then
    WEIGHTS_PATH="$(pwd)/clip-vit-base-patch32.pytorch"
else
    WEIGHTS_PATH="gs://fake-bucket/$MODEL_NAME/weights"
fi

model-zoo deploy "$MODEL_NAME" \
  --weights "$WEIGHTS_PATH" \
  --gpu nvidia-tesla-t4 \
  --target k3d

echo "✓ $MODEL_NAME deployed successfully!"
echo ""
echo "Next steps:"
echo "1. Check status: model-zoo list"
echo "2. Run inference: model-zoo infer $MODEL_NAME --file test_images/fixtures/cat_office_typing.jpg --text 'a cat'"
echo "3. View logs: model-zoo logs $MODEL_NAME --tail"