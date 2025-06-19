#!/bin/bash
set -euo pipefail

# Script to start a local k3d cluster with fake GPU support for development

echo "Starting k3d cluster for local development..."

# Check if k3d is installed
if ! command -v k3d &> /dev/null; then
    echo "k3d not found. Please install k3d first: https://k3d.io/#installation"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "kubectl not found. Please install kubectl first"
    exit 1
fi

# Delete existing cluster if it exists
if k3d cluster list | grep -q "model-zoo-dev"; then
    echo "Deleting existing model-zoo-dev cluster..."
    k3d cluster delete model-zoo-dev
fi

# Create new k3d cluster
echo "Creating k3d cluster 'model-zoo-dev'..."
k3d cluster create model-zoo-dev \
    --agents 2 \
    --port "10001:10001@loadbalancer" \
    --port "8265:8265@loadbalancer" \
    --k3s-arg "--disable=traefik@server:0" \
    --wait

# Wait for cluster to be ready
echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=ready nodes --all --timeout=60s

# Create fake GPU labels for development (since we're on CPU-only)
echo "Adding fake GPU labels for development..."
kubectl label nodes --all nvidia.com/gpu.present=true --overwrite

echo "k3d cluster 'model-zoo-dev' is ready!"
echo ""
echo "Cluster info:"
kubectl cluster-info
echo ""
echo "Nodes:"
kubectl get nodes -o wide