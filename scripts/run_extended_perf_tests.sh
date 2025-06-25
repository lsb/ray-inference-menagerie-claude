#!/bin/bash
set -euo pipefail

# Run extended performance tests for detailed overhead analysis

echo "🚀 Running Extended Performance Tests"
echo "===================================="
echo "These tests will take several minutes to complete"
echo ""

# Check dependencies
echo "1. Checking dependencies..."
python -c "import ray, torch, transformers, pytest" || {
    echo "Missing dependencies. Install with: pip install -e .[dev]"
    exit 1
}

# Check if CLIP weights exist
if [[ ! -f "clip-vit-base-patch32.pytorch" ]]; then
    echo "❌ CLIP weights not found. Downloading..."
    curl -L https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin \
        -o clip-vit-base-patch32.pytorch
fi

# Is-Odd Million Iteration Test
echo ""
echo "2. Running Is-Odd Million Iteration Test"
echo "   This will run 1,000,000 inferences to measure precise Ray overhead"
echo "   Expected time: 2-5 minutes"
echo ""

start_time=$(date +%s)
python -m pytest tests/perf/test_is_odd_million.py::test_is_odd_million_iterations -v -s

is_odd_time=$(($(date +%s) - start_time))
echo "   ✓ Is-Odd test completed in ${is_odd_time}s"

# Generate test image formats if needed
if [[ ! -d "test_images/formats" ]]; then
    echo ""
    echo "3. Generating test image formats..."
    python scripts/generate_test_image_formats.py
fi

# CLIP Comprehensive Performance Test  
echo ""
echo "3. Running CLIP Comprehensive Performance Test"
echo "   This will test multiple image formats with 500 iterations each"
echo "   12 images × 500 iterations × 2 classifications = 12,000 total inferences"
echo "   Expected time: 10-15 minutes"
echo ""

start_time=$(date +%s)
python -m pytest tests/perf/test_clip_comprehensive.py::test_clip_comprehensive -v -s

clip_time=$(($(date +%s) - start_time))
echo "   ✓ CLIP comprehensive test completed in ${clip_time}s"

# Summary
total_time=$((is_odd_time + clip_time))
echo ""
echo "🎉 Extended Performance Tests Complete!"
echo "======================================"
echo "Total time: ${total_time}s"
echo ""
echo "📊 Key Insights:"
echo "   • Is-Odd: Precise Ray overhead measurement over 1M iterations"
echo "   • CLIP: Comprehensive latency analysis over 600+ inferences"
echo "   • Results saved to clip_performance_results.json"
echo ""
echo "💡 Next Steps:"
echo "   • Review latency percentiles for capacity planning"
echo "   • Use throughput metrics for scaling decisions"
echo "   • Compare P99 latencies for SLA definitions"