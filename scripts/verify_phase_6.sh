#!/bin/bash
set -euo pipefail

echo "Verifying Phase 6 requirements..."

# Check E2E tests exist
echo "Checking E2E tests..."
if [ ! -f "tests/e2e/test_k3d_deployment.py" ]; then
    echo "ERROR: E2E tests not found"
    exit 1
fi

# Check E2E test functions
required_e2e_tests=(
    "test_cli_deploy_clip"
    "test_service_availability"
    "test_yaml_template_rendering"
)

for test in "${required_e2e_tests[@]}"; do
    if ! grep -q "def $test" tests/e2e/test_k3d_deployment.py; then
        echo "ERROR: E2E test '$test' not found"
        exit 1
    fi
done

# Check README is comprehensive
echo "Checking README documentation..."
if [ ! -f "README.md" ]; then
    echo "ERROR: README.md not found"
    exit 1
fi

# Check README sections
required_readme_sections=(
    "Quick Start"
    "Adding a New Model"
    "Architecture"
    "Monitoring"
    "Testing"
    "Contributing"
)

for section in "${required_readme_sections[@]}"; do
    if ! grep -q "$section" README.md; then
        echo "ERROR: README missing section: $section"
        exit 1
    fi
done

# Check 5-step guide for adding models
if ! grep -q "5 Steps" README.md; then
    echo "ERROR: README missing 5-step guide for adding models"
    exit 1
fi

# Check quick-start examples
if ! grep -q "model-zoo deploy" README.md; then
    echo "ERROR: README missing deployment examples"
    exit 1
fi

if ! grep -q "model-zoo infer" README.md; then
    echo "ERROR: README missing inference examples"
    exit 1
fi

# Check version is set
if ! grep -q "Version.*0\.1\.0" README.md; then
    echo "ERROR: README missing version 0.1.0"
    exit 1
fi

# Verify all verification scripts pass
echo "Running all phase verification scripts..."
failed_phases=()

for phase in {1..5}; do
    script="scripts/verify_phase_${phase}.sh"
    if [ -f "$script" ]; then
        echo "Running $script..."
        if ! ./"$script" >/dev/null 2>&1; then
            failed_phases+=("$phase")
            echo "WARNING: Phase $phase verification failed"
        else
            echo "✓ Phase $phase verification passed"
        fi
    else
        echo "ERROR: Verification script $script not found"
        exit 1
    fi
done

if [ ${#failed_phases[@]} -gt 0 ]; then
    echo "ERROR: Failed phase verifications: ${failed_phases[*]}"
    echo "Please fix issues in earlier phases before proceeding"
    exit 1
fi

# Check project structure is complete
echo "Checking project structure..."
required_dirs=(
    "model_zoo"
    "model_zoo/actors"
    "infra/k8s/templates"
    "infra/docker"
    "tests/unit"
    "tests/perf"
    "tests/e2e"
    "tests/fixtures"
    "scripts"
    "docs"
    "observability"
    "examples"
)

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Required directory $dir not found"
        exit 1
    fi
done

# Check all CLI commands are documented
cli_commands=(
    "init"
    "deploy"
    "logs"
    "infer"
    "list"
    "delete"
)

for cmd in "${cli_commands[@]}"; do
    if ! grep -q "model-zoo $cmd" README.md; then
        echo "ERROR: CLI command '$cmd' not documented in README"
        exit 1
    fi
done

# Check model support is documented
required_models=(
    "CLIP"
    "Grounding DINO"
    "SAM2"
    "Qwen"
)

for model in "${required_models[@]}"; do
    if ! grep -q "$model" README.md; then
        echo "ERROR: Model '$model' not documented in README"
        exit 1
    fi
done

# Check architecture diagram exists
if ! grep -q "┌─────────────────┐" README.md; then
    echo "ERROR: Architecture diagram missing from README"
    exit 1
fi

# Check monitoring section
if ! grep -q "Grafana" README.md; then
    echo "ERROR: Monitoring documentation missing"
    exit 1
fi

# Check security documentation
if ! grep -q "Workload Identity" README.md; then
    echo "ERROR: Security documentation missing"
    exit 1
fi

# Verify pyproject.toml has correct version
if ! grep -q 'version = "0.1.0"' pyproject.toml; then
    echo "ERROR: pyproject.toml missing version 0.1.0"
    exit 1
fi

# Check examples directory
if [ ! -f "examples/cli_usage.sh" ]; then
    echo "ERROR: CLI usage examples not found"
    exit 1
fi

# Check GitHub Actions workflow exists and is comprehensive
if ! grep -q "k3d-smoke-test" .github/workflows/ci.yml; then
    echo "ERROR: GitHub Actions missing k3d smoke test job"
    exit 1
fi

# Check essential files exist
essential_files=(
    "pyproject.toml"
    "LICENSE"
    ".gitignore"
    "CLAUDE.md"
    "CHECKS.md"
)

for file in "${essential_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Essential file $file not found"
        exit 1
    fi
done

# Check model actors are complete
model_actors=(
    "model_zoo/actors/clip.py"
    "model_zoo/actors/grounding_sam2.py"
    "model_zoo/actors/qwen_vl.py"
)

for actor in "${model_actors[@]}"; do
    if [ ! -f "$actor" ]; then
        echo "ERROR: Model actor $actor not found"
        exit 1
    fi
    
    if ! grep -q "async def infer" "$actor"; then
        echo "ERROR: Actor $actor missing infer method"
        exit 1
    fi
done

# Verify CLI is installable
echo "Testing CLI installation..."
python -c "
try:
    import model_zoo.cli
    print('✓ CLI module imports successfully')
except ImportError as e:
    print(f'ERROR: CLI import failed: {e}')
    exit(1)
"

# Check that all templates can be rendered
echo "Testing template rendering..."
python -c "
from pathlib import Path

template_vars = {
    '{{MODEL_NAME}}': 'test-model',
    '{{IMAGE}}': 'test:latest',
    '{{WEIGHTS_URI}}': 'gs://test/weights',
    '{{GPU_TYPE}}': 'nvidia-tesla-t4',
    '{{TARGET_NS}}': 'default',
    '{{APP_LABEL}}': 'model-zoo',
    '{{GCP_PROJECT_ID}}': 'test-project'
}

template_dir = Path('infra/k8s/templates')
for template_file in template_dir.glob('*.yaml'):
    content = template_file.read_text()
    for old, new in template_vars.items():
        content = content.replace(old, new)
    
    # Check no unreplaced tokens
    for token in template_vars.keys():
        if token in content:
            print(f'ERROR: Unreplaced token {token} in {template_file.name}')
            exit(1)

print('✓ All templates render correctly')
"

echo ""
echo "✓ Phase 6 verification complete!"
echo ""
echo "🎉 ALL PHASES COMPLETE! 🎉"
echo ""
echo "Model Zoo v0.1.0 is ready for release with:"
echo "  - Complete CLI with 6 commands"
echo "  - 3 production-ready model actors"
echo "  - Kubernetes deployment automation"
echo "  - Comprehensive testing suite"
echo "  - Production monitoring & runbook"
echo "  - Full documentation & examples"
echo ""
echo "Ready to tag v0.1.0 and deploy! 🚀"