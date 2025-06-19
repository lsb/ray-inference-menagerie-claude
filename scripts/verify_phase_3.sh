#!/bin/bash
set -euo pipefail

echo "Verifying Phase 3 requirements..."

# Check model actor implementations
echo "Checking model actor implementations..."
required_actors=(
    "model_zoo/actors/clip.py"
    "model_zoo/actors/grounding_sam2.py"
    "model_zoo/actors/qwen_vl.py"
)

for actor in "${required_actors[@]}"; do
    if [ ! -f "$actor" ]; then
        echo "ERROR: Required actor $actor does not exist"
        exit 1
    fi
    
    # Check actor inherits from HFModelActor
    if ! grep -q "class.*HFModelActor" "$actor"; then
        echo "ERROR: $actor does not inherit from HFModelActor"
        exit 1
    fi
    
    # Check required methods
    if ! grep -q "async def _load_model" "$actor"; then
        echo "ERROR: $actor missing _load_model method"
        exit 1
    fi
    
    if ! grep -q "async def infer" "$actor"; then
        echo "ERROR: $actor missing infer method"
        exit 1
    fi
done

# Check CLIP actor specifics
echo "Validating CLIP actor..."
if ! grep -q '"similarity"' model_zoo/actors/clip.py; then
    echo "ERROR: CLIP actor doesn't return similarity score"
    exit 1
fi

# Check Grounding DINO + SAM2 actor specifics
echo "Validating Grounding DINO + SAM2 actor..."
if ! grep -q '"mask_png_b64"' model_zoo/actors/grounding_sam2.py; then
    echo "ERROR: Grounding DINO + SAM2 actor doesn't return mask_png_b64"
    exit 1
fi

# Check Qwen VL actor specifics
echo "Validating Qwen VL actor..."
if ! grep -q '"answer"' model_zoo/actors/qwen_vl.py; then
    echo "ERROR: Qwen VL actor doesn't return answer"
    exit 1
fi

# Check test fixtures
echo "Checking test fixtures..."
if [ ! -d "tests/fixtures" ]; then
    echo "ERROR: tests/fixtures directory does not exist"
    exit 1
fi

if [ ! -f "tests/fixtures/test_image.py" ]; then
    echo "ERROR: test_image.py fixture not found"
    exit 1
fi

# Check performance tests
echo "Checking performance tests..."
if [ ! -f "tests/perf/test_canaries.py" ]; then
    echo "ERROR: test_canaries.py not found"
    exit 1
fi

# Verify test structure
if ! grep -q "test_clip_actor_performance" tests/perf/test_canaries.py; then
    echo "ERROR: CLIP performance test not found"
    exit 1
fi

if ! grep -q "test_grounding_sam2_actor_performance" tests/perf/test_canaries.py; then
    echo "ERROR: Grounding DINO + SAM2 performance test not found"
    exit 1
fi

if ! grep -q "test_qwen_vl_actor_performance" tests/perf/test_canaries.py; then
    echo "ERROR: Qwen VL performance test not found"
    exit 1
fi

# Check latency assertions
if ! grep -q "assert.*< 5" tests/perf/test_canaries.py; then
    echo "ERROR: Performance tests missing latency < 5s assertion"
    exit 1
fi

# Run a simple Python import test
echo "Testing Python imports..."
python -c "
try:
    from model_zoo.actors.clip import CLIPActor
    from model_zoo.actors.grounding_sam2 import GroundingDINO_SAM2_Actor
    from model_zoo.actors.qwen_vl import QwenVLActor
    from tests.fixtures import TEST_IMAGE_B64
    print('✓ All imports successful')
except ImportError as e:
    print(f'ERROR: Import failed: {e}')
    exit(1)
"

echo ""
echo "✓ Phase 3 verification complete!"
echo "All model actors implemented with test fixtures and performance tests."