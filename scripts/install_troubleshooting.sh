#!/bin/bash
set -euo pipefail

# Troubleshooting script for installation issues

echo "🔧 Model Zoo Installation Troubleshooting"
echo "=========================================="

echo "1. Checking Python version..."
python --version

PYTHON_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
if [[ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.10" ]]; then
    echo "❌ Python 3.10+ required, found $PYTHON_VERSION"
    echo "Solutions:"
    echo "- Install Python 3.10+: https://python.org/downloads/"
    echo "- Try: python3.10 -m pip install -e ."
    echo "- Try: python3.11 -m pip install -e ."
    exit 1
else
    echo "✅ Python version OK: $PYTHON_VERSION"
fi

echo ""
echo "2. Attempting standard installation..."
if pip install -e .; then
    echo "✅ Standard installation successful!"
    exit 0
fi

echo "❌ Standard installation failed. Trying alternatives..."

echo ""
echo "3. Alternative 1: PYTHONPATH method..."
pip install ray[default] typer rich kubernetes pillow transformers torch pytest-asyncio
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
if python -m model_zoo.cli --help &>/dev/null; then
    echo "✅ PYTHONPATH method works!"
    echo "Add this to your shell profile:"
    echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\""
    exit 0
fi

echo ""
echo "4. Alternative 2: Minimal install..."
pip install ray[default] typer rich pillow
if python -c "from model_zoo.actors.clip import CLIPActor; print('✅ Basic import works')"; then
    echo "✅ Minimal install works for basic testing"
    exit 0
fi

echo ""
echo "❌ All installation methods failed. Please:"
echo "1. Check your Python environment"
echo "2. Try creating a fresh virtual environment"
echo "3. Report the issue with full error logs"