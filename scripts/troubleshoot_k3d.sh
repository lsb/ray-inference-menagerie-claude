#!/bin/bash
set -euo pipefail

# Troubleshooting script for k3d deployment issues

echo "🔍 K3d Troubleshooting Script"
echo "============================="

# Check pod status
echo "1. Checking pod status..."
kubectl get pods -A

echo ""
echo "2. Checking services..."
kubectl get svc

echo ""
echo "3. Checking nodes..."
kubectl get nodes -o wide

echo ""
echo "4. Checking detailed logs..."
kubectl logs -l app=model-zoo --tail=50

echo ""
echo "5. Verifying image imports..."
if k3d cluster list | grep -q "model-zoo-dev"; then
    docker exec k3d-model-zoo-dev-agent-0 crictl images | grep model-zoo || echo "No model-zoo images found"
else
    echo "model-zoo-dev cluster not found"
fi

echo ""
echo "6. Checking cluster info..."
kubectl cluster-info

echo ""
echo "🔧 Common fixes:"
echo "- Restart cluster: k3d cluster delete model-zoo-dev && ./scripts/dev_cluster.sh"
echo "- Rebuild image: docker build -t model-zoo-clip-test:latest -f infra/docker/Dockerfile.model ."
echo "- Re-import image: k3d image import model-zoo-clip-test:latest -c model-zoo-dev"
echo "- Manual port-forward: kubectl port-forward svc/clip-test-ray-head 10001:10001 8265:8265"