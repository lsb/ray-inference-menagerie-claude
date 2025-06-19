#!/bin/bash
set -euo pipefail

echo "Verifying Phase 5 requirements..."

# Check GitHub Actions matrix build
echo "Checking GitHub Actions CI/CD pipeline..."
if [ ! -f ".github/workflows/ci.yml" ]; then
    echo "ERROR: GitHub Actions workflow not found"
    exit 1
fi

# Check for matrix build strategy
if ! grep -q "strategy:" .github/workflows/ci.yml; then
    echo "ERROR: Matrix build strategy not found"
    exit 1
fi

if ! grep -q "matrix:" .github/workflows/ci.yml; then
    echo "ERROR: Matrix configuration not found"
    exit 1
fi

# Check for build job
if ! grep -q "build:" .github/workflows/ci.yml; then
    echo "ERROR: Build job not found"
    exit 1
fi

# Check for k3d smoke test
if ! grep -q "k3d-smoke-test:" .github/workflows/ci.yml; then
    echo "ERROR: k3d smoke test job not found"
    exit 1
fi

# Check for manual approval gate
if ! grep -q "approve-prod:" .github/workflows/ci.yml; then
    echo "ERROR: Manual approval gate not found"
    exit 1
fi

if ! grep -q "environment: production" .github/workflows/ci.yml; then
    echo "ERROR: Production environment gate not configured"
    exit 1
fi

# Check enhanced HPA template
echo "Checking HPA autoscaling configuration..."
if ! grep -q "ray_actor_queue_size" infra/k8s/templates/hpa.yaml; then
    echo "ERROR: HPA missing ray_actor_queue_size metric"
    exit 1
fi

if ! grep -q "External" infra/k8s/templates/hpa.yaml; then
    echo "ERROR: HPA missing external metrics"
    exit 1
fi

if ! grep -q "behavior:" infra/k8s/templates/hpa.yaml; then
    echo "ERROR: HPA missing scaling behavior configuration"
    exit 1
fi

if ! grep -q "minReplicas: 0" infra/k8s/templates/hpa.yaml; then
    echo "ERROR: HPA doesn't support scale-to-zero"
    exit 1
fi

# Check production runbook
echo "Checking production documentation..."
if [ ! -f "docs/production_runbook.md" ]; then
    echo "ERROR: Production runbook not found"
    exit 1
fi

# Check runbook sections
required_sections=(
    "## Overview"
    "## Deployment Procedures"
    "## Monitoring & Observability" 
    "## Troubleshooting"
    "## Emergency Procedures"
    "## Security Considerations"
)

for section in "${required_sections[@]}"; do
    if ! grep -q "$section" docs/production_runbook.md; then
        echo "ERROR: Runbook missing section: $section"
        exit 1
    fi
done

# Check Grafana dashboard
echo "Checking Grafana dashboard..."
if [ ! -f "observability/grafana_model_zoo.json" ]; then
    echo "ERROR: Grafana dashboard not found"
    exit 1
fi

# Check dashboard has essential panels
if ! grep -q "Ray Actor Queue Size" observability/grafana_model_zoo.json; then
    echo "ERROR: Dashboard missing Ray Actor Queue Size panel"
    exit 1
fi

if ! grep -q "GPU Utilization" observability/grafana_model_zoo.json; then
    echo "ERROR: Dashboard missing GPU Utilization panel"
    exit 1
fi

if ! grep -q "Inference Latency" observability/grafana_model_zoo.json; then
    echo "ERROR: Dashboard missing Inference Latency panel"
    exit 1
fi

# Check service account template
echo "Checking service account and RBAC..."
if [ ! -f "infra/k8s/templates/serviceaccount.yaml" ]; then
    echo "ERROR: Service account template not found"
    exit 1
fi

if ! grep -q "iam.gke.io/gcp-service-account" infra/k8s/templates/serviceaccount.yaml; then
    echo "ERROR: Workload Identity annotation missing"
    exit 1
fi

if ! grep -q "kind: Role" infra/k8s/templates/serviceaccount.yaml; then
    echo "ERROR: RBAC Role not found"
    exit 1
fi

if ! grep -q "kind: RoleBinding" infra/k8s/templates/serviceaccount.yaml; then
    echo "ERROR: RBAC RoleBinding not found"
    exit 1
fi

# Check CLI supports GCP project ID
if ! grep -q "GCP_PROJECT_ID" model_zoo/cli.py; then
    echo "ERROR: CLI doesn't support GCP_PROJECT_ID environment variable"
    exit 1
fi

# Verify Docker buildx in CI
if ! grep -q "docker/setup-buildx-action" .github/workflows/ci.yml; then
    echo "ERROR: Docker Buildx not configured in CI"
    exit 1
fi

# Check for image caching
if ! grep -q "cache-from: type=gha" .github/workflows/ci.yml; then
    echo "ERROR: GitHub Actions cache not configured"
    exit 1
fi

# Check security: no secrets in repo
echo "Checking security configuration..."
if grep -r "password\|secret\|key" --include="*.py" --include="*.yaml" --include="*.yml" . | grep -v "# " | grep -v password_hash; then
    echo "WARNING: Potential secrets found in code (review above)"
fi

# Check LoadBalancer security
if ! grep -q "loadBalancerSourceRanges" infra/k8s/templates/service.yaml; then
    echo "ERROR: LoadBalancer not restricted by source IP ranges"
    exit 1
fi

echo ""
echo "✓ Phase 5 verification complete!"
echo "CI/CD pipeline, autoscaling, monitoring, and security configurations verified."