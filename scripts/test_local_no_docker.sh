#!/bin/bash
set -euo pipefail

# Simple local test without Docker/k3d - just pure Ray actors

echo "🧪 Testing Model Zoo locally without Docker/k3d"
echo "==============================================="

# Check dependencies
echo "1. Checking dependencies..."
python -c "import ray, torch, transformers" || {
    echo "Missing dependencies. Install with: pip install -e ."
    exit 1
}

# Check if CLIP weights exist
if [[ ! -f "clip-vit-base-patch32.pytorch" ]]; then
    echo "❌ CLIP weights not found. Downloading..."
    curl -L https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin \
        -o clip-vit-base-patch32.pytorch
fi

echo "✅ CLIP weights found ($(du -h clip-vit-base-patch32.pytorch | cut -f1))"

# Run E2E tests with Ray actors
echo ""
echo "2. Running E2E Ray actor tests..."
python -m pytest tests/e2e/test_e2e_local.py::test_clip_actor_e2e -v -s

echo ""
echo "3. Running all model types test..."
python -m pytest tests/e2e/test_e2e_all_models.py::test_all_models_e2e -v -s

echo ""
echo "🎉 Local Ray actor tests complete!"
echo "✅ CLIP works with local weights"
echo "✅ All 3 model types functional"
echo "✅ Concurrent inference working"
echo "✅ Horizontal scaling validated"