# Phase Completion Checklist

## Phase 1 - Repo Skeleton & CI Stub ✅

- [x] Created `model_zoo/` Python package with `__init__.py`
- [x] Created `infra/k8s/templates/` with blank YAML files:
  - [x] `head.yaml`
  - [x] `worker.yaml` 
  - [x] `service.yaml`
  - [x] `hpa.yaml`
- [x] Created `infra/docker/Dockerfile.base-pygpu` (CUDA 12, Python 3.11, uv)
- [x] Created `scripts/dev_cluster.sh` for k3d with fake GPU support
- [x] Created GitHub Actions `.github/workflows/ci.yml` (ruff, pytest)
- [x] Created `scripts/verify_phase_1.sh` to validate YAML templates
- [x] Created `pyproject.toml` (from previous commit)

Completed: 2025-06-19

## Phase 2 - YAML Templates & Ray Actor Base

- [ ] Fill YAML templates with placeholder tokens
- [ ] Create head deployment with ray-head and driver containers
- [ ] Create worker deployment with GPU support
- [ ] Create service manifest
- [ ] Create HPA manifest
- [ ] Implement `model_zoo/actors/base.py` with HFModelActor
- [ ] Implement `model_zoo/driver.py`
- [ ] Create `scripts/verify_phase_2.sh`

## Phase 3 - Canonical Model Actors & Synthetic Data

- [ ] Implement CLIPActor
- [ ] Implement GroundingDINO_SAM2_Actor
- [ ] Implement QwenVLActor
- [ ] Add test fixtures
- [ ] Create performance tests
- [ ] Create `scripts/verify_phase_3.sh`

## Phase 4 - CLI (Typer) & Developer UX

- [ ] Implement CLI commands
- [ ] Build and push workflow
- [ ] YAML rendering
- [ ] Deployment automation
- [ ] Canary testing
- [ ] Create `scripts/verify_phase_4.sh`

## Phase 5 - CI/CD & Autoscaling

- [ ] GitHub Actions matrix builds
- [ ] Manual approval gates
- [ ] HPA template generation
- [ ] Production runbook
- [ ] Grafana dashboard
- [ ] Create `scripts/verify_phase_5.sh`

## Phase 6 - Smoke Tests & Release

- [ ] E2E tests with k3d
- [ ] Update README
- [ ] Tag v0.1.0
- [ ] Create `scripts/verify_phase_6.sh`