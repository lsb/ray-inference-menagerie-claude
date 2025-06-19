#!/bin/bash
set -euo pipefail

echo "Verifying Model Zoo installation..."

# Check Python version
echo "Checking Python version..."
python -c "
import sys
if sys.version_info >= (3, 10):
    print(f'✓ Python {sys.version.split()[0]} (>= 3.10 required)')
else:
    print(f'✗ Python {sys.version.split()[0]} (>= 3.10 required)')
    exit(1)
"

# Test Python imports
echo "Testing Python imports..."
python -c "
import sys
sys.path.insert(0, '.')

try:
    import model_zoo
    print('✓ model_zoo imports')
    
    import model_zoo.cli
    print('✓ model_zoo.cli imports')
    
    from model_zoo.actors.base import HFModelActor
    print('✓ HFModelActor imports')
    
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"

# Test CLI functionality
echo "Testing CLI help..."
python -m model_zoo.cli --help >/dev/null 2>&1 && echo "✓ CLI help works" || echo "✗ CLI help failed"

# Check required dependencies
echo "Checking key dependencies..."
python -c "
deps = ['ray', 'typer', 'rich', 'pillow']
missing = []

for dep in deps:
    try:
        __import__(dep)
        print(f'✓ {dep} available')
    except ImportError:
        missing.append(dep)
        print(f'✗ {dep} missing')

if missing:
    print(f'Missing dependencies: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
    exit(1)
else:
    print('✓ All key dependencies available')
"

echo ""
echo "✅ Installation verification complete!"
echo "Ready to deploy models with: python -m model_zoo.cli"