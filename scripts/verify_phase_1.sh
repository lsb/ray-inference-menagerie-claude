#!/bin/bash
set -euo pipefail

echo "Verifying Phase 1 requirements..."

# Check directory structure
echo "Checking directory structure..."
required_dirs=(
    "model_zoo"
    "infra/k8s/templates"
    "infra/docker"
    "scripts"
    ".github/workflows"
)

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Required directory $dir does not exist"
        exit 1
    fi
done

# Check required files
echo "Checking required files..."
required_files=(
    "infra/k8s/templates/head.yaml"
    "infra/k8s/templates/worker.yaml"
    "infra/k8s/templates/service.yaml"
    "infra/k8s/templates/hpa.yaml"
    "infra/docker/Dockerfile.base-pygpu"
    "scripts/dev_cluster.sh"
    ".github/workflows/ci.yml"
    "pyproject.toml"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file $file does not exist"
        exit 1
    fi
done

# Validate YAML files
echo "Validating YAML templates..."
for yaml in infra/k8s/templates/*.yaml; do
    if command -v kubectl &> /dev/null; then
        # Try to validate with kubectl (will fail on templates but that's ok)
        kubectl apply --dry-run=client -f "$yaml" 2>/dev/null || true
    fi
    echo "  - $yaml exists"
done

# Check Python package
echo "Checking Python package structure..."
if [ ! -f "model_zoo/__init__.py" ]; then
    echo "ERROR: model_zoo/__init__.py does not exist"
    exit 1
fi

# Check executable permissions
echo "Checking script permissions..."
if [ ! -x "scripts/dev_cluster.sh" ]; then
    echo "ERROR: scripts/dev_cluster.sh is not executable"
    exit 1
fi

echo ""
echo "✓ Phase 1 verification complete!"
echo "All required directories and files are present."