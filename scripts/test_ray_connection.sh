#!/bin/bash
set -euo pipefail

# Test Ray connection directly

echo "Testing Ray connection..."

# Test Ray connection directly
python -c "
import ray
try:
    ray.init('ray://localhost:10001')
    print('✓ Connected to Ray at localhost:10001')
    print('Dashboard: http://localhost:8265')
    ray.shutdown()
except Exception as e:
    print(f'✗ Failed to connect to Ray: {e}')
    print('Make sure Ray cluster is running and port 10001 is accessible')
    exit(1)
"

echo "✓ Ray connection test complete!"