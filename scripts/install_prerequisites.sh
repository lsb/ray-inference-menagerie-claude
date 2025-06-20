#!/bin/bash
set -euo pipefail

# Install prerequisites for local development

echo "Installing prerequisites for Model Zoo development..."

# Check OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    
    # Install k3d and kubectl via Homebrew
    if command -v brew &> /dev/null; then
        echo "Installing k3d and kubectl via Homebrew..."
        brew install k3d kubectl
    else
        echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    
    # Install k3d
    echo "Installing k3d..."
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
    
    # Install kubectl
    echo "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/
    
else
    echo "Unsupported OS: $OSTYPE"
    echo "Please install k3d and kubectl manually:"
    echo "- k3d: https://k3d.io/#installation"
    echo "- kubectl: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# Verify installations
echo "Verifying installations..."
k3d --version
kubectl version --client

echo "✓ Prerequisites installed successfully!"
echo ""
echo "Next steps:"
echo "1. Install Model Zoo: pip install -e ."
echo "2. Start local cluster: ./scripts/dev_cluster.sh"