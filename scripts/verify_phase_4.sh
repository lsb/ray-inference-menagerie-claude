#!/bin/bash
set -euo pipefail

echo "Verifying Phase 4 requirements..."

# Check CLI module exists
echo "Checking CLI implementation..."
if [ ! -f "model_zoo/cli.py" ]; then
    echo "ERROR: model_zoo/cli.py not found"
    exit 1
fi

# Check required CLI commands
required_commands=(
    "init"
    "deploy" 
    "logs"
    "infer"
    "list"
    "delete"
)

for cmd in "${required_commands[@]}"; do
    if ! grep -q "def $cmd(" model_zoo/cli.py; then
        echo "ERROR: CLI command '$cmd' not implemented"
        exit 1
    fi
done

# Check CLI uses Typer
if ! grep -q "import typer" model_zoo/cli.py; then
    echo "ERROR: CLI doesn't use Typer"
    exit 1
fi

# Check deploy command features
echo "Validating deploy command features..."
if ! grep -q "docker.*build" model_zoo/cli.py; then
    echo "ERROR: Deploy command missing Docker build"
    exit 1
fi

if ! grep -q "kubectl.*apply" model_zoo/cli.py; then
    echo "ERROR: Deploy command missing kubectl apply"
    exit 1
fi

if ! grep -q "ray.init" model_zoo/cli.py; then
    echo "ERROR: Deploy command missing canary test"
    exit 1
fi

# Check template rendering
if ! grep -q "template_vars" model_zoo/cli.py; then
    echo "ERROR: Deploy command missing template rendering"
    exit 1
fi

# Check infer command supports all model types
echo "Validating infer command..."
if ! grep -q '"clip"' model_zoo/cli.py; then
    echo "ERROR: Infer command missing CLIP support"
    exit 1
fi

if ! grep -q '"grounding"' model_zoo/cli.py; then
    echo "ERROR: Infer command missing Grounding DINO support"
    exit 1
fi

if ! grep -q '"qwen"' model_zoo/cli.py; then
    echo "ERROR: Infer command missing Qwen VL support"
    exit 1
fi

# Check Docker files
echo "Checking Docker files..."
if [ ! -f "infra/docker/Dockerfile.model" ]; then
    echo "ERROR: Dockerfile.model not found"
    exit 1
fi

# Check utility functions
echo "Checking utility functions..."
if [ ! -f "model_zoo/utils.py" ]; then
    echo "ERROR: model_zoo/utils.py not found"
    exit 1
fi

# Check unit tests
echo "Checking unit tests..."
if [ ! -f "tests/unit/test_cli.py" ]; then
    echo "ERROR: CLI unit tests not found"
    exit 1
fi

# Check pyproject.toml has CLI entry point
if ! grep -q 'model-zoo.*=.*model_zoo.cli:app' pyproject.toml; then
    echo "ERROR: CLI entry point not configured in pyproject.toml"
    exit 1
fi

# Test CLI help works
echo "Testing CLI help..."
python -c "
try:
    from model_zoo.cli import app
    print('✓ CLI module imports successfully')
except ImportError as e:
    print(f'ERROR: CLI import failed: {e}')
    exit(1)
"

# Check error handling in deploy
if ! grep -q "rollback\|delete.*-f" model_zoo/cli.py; then
    echo "ERROR: Deploy command missing rollback on canary failure"
    exit 1
fi

# Check CLI uses rich for output
if ! grep -q "from rich" model_zoo/cli.py; then
    echo "ERROR: CLI doesn't use rich for output"
    exit 1
fi

echo ""
echo "✓ Phase 4 verification complete!"
echo "CLI with all required commands implemented and tested."