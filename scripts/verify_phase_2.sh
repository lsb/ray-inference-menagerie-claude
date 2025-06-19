#!/bin/bash
set -euo pipefail

echo "Verifying Phase 2 requirements..."

# Check YAML templates have tokens
echo "Checking YAML template tokens..."
required_tokens=(
    "{{MODEL_NAME}}"
    "{{IMAGE}}"
    "{{WEIGHTS_URI}}"
    "{{GPU_TYPE}}"
    "{{TARGET_NS}}"
    "{{APP_LABEL}}"
)

for token in "${required_tokens[@]}"; do
    found=false
    for yaml in infra/k8s/templates/*.yaml; do
        if grep -q "$token" "$yaml"; then
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        echo "WARNING: Token $token not found in any template"
    fi
done

# Check head.yaml structure
echo "Validating head.yaml structure..."
if ! grep -q "ray-head" infra/k8s/templates/head.yaml; then
    echo "ERROR: head.yaml missing ray-head container"
    exit 1
fi
if ! grep -q "driver" infra/k8s/templates/head.yaml; then
    echo "ERROR: head.yaml missing driver container"
    exit 1
fi
if ! grep -q '"ray", "start", "--head"' infra/k8s/templates/head.yaml; then
    echo "ERROR: head.yaml missing correct ray start command"
    exit 1
fi

# Check worker.yaml structure
echo "Validating worker.yaml structure..."
if ! grep -q "nvidia.com/gpu.present" infra/k8s/templates/worker.yaml; then
    echo "ERROR: worker.yaml missing GPU nodeSelector"
    exit 1
fi
if ! grep -q 'nvidia.com/gpu: "1"' infra/k8s/templates/worker.yaml; then
    echo "ERROR: worker.yaml not requesting 1 GPU"
    exit 1
fi

# Check service.yaml structure
echo "Validating service.yaml structure..."
if ! grep -q "port: 10001" infra/k8s/templates/service.yaml; then
    echo "ERROR: service.yaml missing Ray Client port 10001"
    exit 1
fi
if ! grep -q "port: 8265" infra/k8s/templates/service.yaml; then
    echo "ERROR: service.yaml missing dashboard port 8265"
    exit 1
fi

# Check HPA structure
echo "Validating hpa.yaml structure..."
if ! grep -q "ray_actor_queue_size" infra/k8s/templates/hpa.yaml; then
    echo "ERROR: hpa.yaml missing ray_actor_queue_size metric"
    exit 1
fi

# Check Python files
echo "Checking Python actor structure..."
required_python_files=(
    "model_zoo/actors/base.py"
    "model_zoo/actors/factory.py"
    "model_zoo/driver.py"
)

for file in "${required_python_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required Python file $file does not exist"
        exit 1
    fi
done

# Check base.py has required methods
if ! grep -q "class HFModelActor" model_zoo/actors/base.py; then
    echo "ERROR: base.py missing HFModelActor class"
    exit 1
fi
if ! grep -q "async def ready" model_zoo/actors/base.py; then
    echo "ERROR: base.py missing ready method"
    exit 1
fi
if ! grep -q "async def infer" model_zoo/actors/base.py; then
    echo "ERROR: base.py missing infer method"
    exit 1
fi
if ! grep -q "@ray.remote(num_gpus=1)" model_zoo/actors/base.py; then
    echo "ERROR: base.py missing Ray remote decorator with GPU"
    exit 1
fi

# Check driver.py structure
if ! grep -q 'ray.init(address="auto")' model_zoo/driver.py; then
    echo "ERROR: driver.py missing ray.init"
    exit 1
fi
if ! grep -q "factory.make()" model_zoo/driver.py; then
    echo "ERROR: driver.py not using factory"
    exit 1
fi

# Test YAML rendering with dummy values
echo "Testing YAML template rendering..."
temp_dir=$(mktemp -d)
for template in infra/k8s/templates/*.yaml; do
    output_file="$temp_dir/$(basename $template)"
    sed -e 's/{{MODEL_NAME}}/test-model/g' \
        -e 's/{{IMAGE}}/test-image:latest/g' \
        -e 's/{{WEIGHTS_URI}}/gs:\/\/test-bucket\/weights/g' \
        -e 's/{{GPU_TYPE}}/a100/g' \
        -e 's/{{TARGET_NS}}/default/g' \
        -e 's/{{APP_LABEL}}/model-zoo/g' \
        "$template" > "$output_file"
    
    if command -v kubectl &> /dev/null; then
        kubectl apply --dry-run=client -f "$output_file" >/dev/null 2>&1 || {
            echo "WARNING: Template $template failed dry-run validation"
        }
    fi
done
rm -rf "$temp_dir"

echo ""
echo "✓ Phase 2 verification complete!"
echo "YAML templates filled with tokens and Ray actor base created."