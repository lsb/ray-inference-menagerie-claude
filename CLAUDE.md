# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a model zoo system that deploys cutting-edge ML models (CLIP, Grounding DINO + SAM2, and Qwen 2.5 VL) using Ray on Kubernetes. The system supports both local development (k3d) and cloud deployments (GKE).

## Key Architecture Decisions

1. **Ray Client Protocol**: Inference is reached exclusively through Ray Client protocol (gRPC on port 10001) - there is NO HTTP gateway or Ray Serve
2. **One GPU per Actor**: Each Ray actor uses exactly one GPU; scaling is horizontal only
3. **Cluster per Model**: Each deployment spins up its own Ray cluster via Helm
4. **GCS Weight Storage**: All model weights are stored in GCS with hierarchical layout: `gs://<bucket>/<project>/<model>/<variant>/<sha>/...`
5. **Autoscaling**: Based on `ray_actor_queue_size{name="<model>"} ≤ 1` metric; scale-to-zero allowed

## Development Commands

### Local Development
```bash
# Start local k3d cluster with GPU support
scripts/dev_cluster.sh

# Run tests
pytest -q

# Lint code
ruff check .
ruff format .
```

### CLI Commands (once implemented)
```bash
# Initialize a model family
model-zoo init <model-family> --target gke|k3d

# Deploy a model
model-zoo deploy <model-name> --weights <gcs://...> --gpu a100 --target gke|k3d

# View logs
model-zoo logs <model-name> --tail

# Run inference
model-zoo infer <model-name> --file img.jpg --question "..."
```

## Project Structure

- `model_zoo/` - Core Python package with model actors and drivers
- `infra/helm/` - Helm charts for Kubernetes deployments
- `infra/docker/` - Dockerfiles (base image: CUDA 12, Python 3.11)
- `scripts/` - Development and verification scripts
- `tests/` - Unit tests, performance tests, and test fixtures
- `docs/` - Production documentation
- `observability/` - Monitoring dashboards

## Implementation Phases

The project follows a 6-phase implementation plan (see INSTRUCTIONS.md):
1. Repository scaffolding and CI setup
2. Helm chart and Ray actor template
3. Model actor implementations
4. CLI and developer workflow
5. CI/CD hardening and autoscaling
6. Integration tests and documentation

## Critical Constraints

- **No HTTP gateway** - Use Ray Client protocol only
- **GCP Workload Identity** for authentication (no secrets in repo)
- **Manual promotion gate** in CI/CD pipeline
- **Canary inference** required on every deployment (automatic rollback on failure)
- **Roll-forward only** policy (no version rollbacks)

## Model Actors

Each model actor must:
- Inherit from `HFModelActor` base class
- Implement `async def infer(self, payload: dict) -> dict`
- Implement `async def ready(self) -> bool` for health checks
- Use exactly 1 GPU (`@ray.remote(num_gpus=1)`)

Expected inference response formats:
- **CLIP**: `{similarity: float}`
- **Grounding DINO + SAM2**: `{mask_png_b64: str}`
- **Qwen 2.5 VL**: `{answer: str}`