> I’m trying to make a model zoo for some machine learning models. I want to run some CLIP, Grounding DINO + SAM2, and Qwen 2.5 VL for some visual question answering; implementations in Huggingface are fine. I want to use Ray to deploy arbitrary cutting edge ML research models to Kubernetes. I’m going to want to specify something like what GPU to use with nodeSelectors and annotations but otherwise I’m fine with defaults. We will be storing fine-tunes and we have k8s set up so that with the correct metadata annotation for GCP service account and with the right service account name we can automatically access a particular GCS bucket where the fine tunes will live. We want a command line interface for a team to use to each be able to develop and deploy models independently. We will run in GCP on Kubernetes and locally (via minikube/k3s/k3d/whatever makes sense).

---

## PHASE 0 – Bootstrap & Contract

```
You are an autonomous coding agent.  
Your mission is to build a production-grade “model-zoo” repo that deploys
Hugging Face CLIP, Grounding DINO + SAM2, and Qwen 2.5 VL on raw Ray
(running on Kubernetes).  Inference is reached exclusively through the
**Ray Client** protocol (gRPC on :10001); there is **no HTTP gateway**.

### Hard constraints (MUST):

1. Local Kubernetes = **k3d**; cloud = **GKE** (standard or Autopilot).
2. GPU nodes are modern NVIDIA (T4/A100/L4, some 80 GB).  
   *One Ray actor ⇢ one GPU*; horizontal scaling only.
3. Each deploy spins up its **own Ray cluster** (head + workers) described
   with **raw YAML manifests** (no Helm, no Kustomize; simple templates
   with `{{TOKEN}}` placeholders the CLI will fill).
4. All weights live in one GCS bucket, hierarchical layout  
   `gs://<bucket>/<project>/<model>/<variant>/<sha>/…` (DVC-style hash dir).  
   Ray workers **always** pull weights at start via GCSFuse.
5. Autoscaling goal: keep
   `ray_actor_queue_size{name="<model>"}` ≤ 1; scale-to-zero allowed
   via a Kubernetes **HPA** manifest generated alongside the other YAML.
6. Every cluster’s head pod exposes
   * **10001** – Ray Client  
   * **8265** – Ray Dashboard (optional)  
   and is fronted by a **Service** (LoadBalancer on GKE, ClusterIP on k3d).
7. Logging: stdout; metrics: default Ray + custom latency histogram.
8. CI = GitHub Actions → build images, push to **GCP Artifact Registry**;
   manual “promote” gate; roll-**forward** only.
9. Security: GCP Workload Identity; no secrets in repo; Service has
   `loadBalancerSourceRanges` limited to VPN CIDR.
10. CLI must build/push, **render YAML**, `kubectl apply -f`, wait for
    Service, run a Ray-Client canary inference, and **fail hard** (and
    `kubectl delete -f`) on error.
11. Each usage branch may pin any Ray minor version.

### Assumptions (MAY):

- No PII/HIPAA; model licences out of scope.
- Fine-tune retraining runs in separate clusters.

### Phase deliverables:

After **every** phase you MUST:
1. Commit new code / YAML / docs.  
2. Update `CHECKS.md`.  
3. Provide `scripts/verify_phase_<N>.sh` (exit 0 = success).  
4. Ensure `pytest -q` passes.

Respond **“ACK PHASE 0”** when ready.
```

---

## PHASE 1 – Repo Skeleton & CI Stub

*Goal*: directory tree + tooling.

Checklist

* `model_zoo/` Python package.
* `infra/k8s/templates/` with blank `head.yaml`, `worker.yaml`, `service.yaml`, `hpa.yaml`.
* `infra/docker/Dockerfile.base-pygpu` (CUDA 12, Python 3.11, uv/poetry).
* `scripts/dev_cluster.sh` (k3d, fake-GPU if on CPU-only laptop).
* GitHub `ci.yml` (ruff, pytest).
* `verify_phase_1.sh` (lint YAML with `kubeconform` or `kubectl apply --dry-run=client`).

Return **“DONE PHASE 1”** with diffstat.

---

## PHASE 2 – YAML Templates & Ray Actor Base

*Goal*: fill the template YAML and create the Ray actor scaffold.

### YAML templates (`infra/k8s/templates/`)

* Use `{{MODEL_NAME}}`, `{{IMAGE}}`, `{{WEIGHTS_URI}}`, `{{GPU_TYPE}}`,
  `{{TARGET_NS}}`, `{{APP_LABEL}}` tokens.
* Head Deployment:

  ```yaml
  containers:
  - name: ray-head
    image: {{IMAGE}}
    command: ["ray", "start", "--head",
              "--port=6379",
              "--dashboard-host=0.0.0.0",
              "--num-gpus=0"]
  - name: driver
    image: {{IMAGE}}
    command: ["python", "-m", "model_zoo.driver"]
  ```
* Worker Deployment (1 GPU per pod, nodeSelector on `nvidia.com/gpu.present=true`).
* Service (LoadBalancer / ClusterIP).
* HPA object (cpu-or custom metric).

### Python

`model_zoo/actors/base.py`

```python
@ray.remote(num_gpus=1)
class HFModelActor:
    async def ready(self) -> bool: ...
    async def infer(self, payload: dict) -> dict: ...
```

`model_zoo/driver.py`

```python
import ray, os, time
from model_zoo.actors import factory
ray.init(address="auto")
actor = factory.make().options(
    name=os.environ["MODEL_NAME"],
    num_gpus=1).remote(...)
ray.get(actor.ready.remote())
while True: time.sleep(30)    # keep driver alive
```

`verify_phase_2.sh`

* renders YAML with placeholder values, `kubectl apply --dry-run=client -f -`.

Return **“DONE PHASE 2”.**

---

## PHASE 3 – Canonical Model Actors & Synthetic Data

Implement `CLIPActor`, `GroundingDINO_SAM2_Actor`, `QwenVLActor`
(subclassing `HFModelActor`).
Add fixtures under `tests/fixtures/`.
`tests/perf/test_canaries.py` spins up a local Ray head, instantiates each
actor, calls `infer()`, and asserts success & latency < 5 s (CPU).

Return **“DONE PHASE 3”.**

---

## PHASE 4 – CLI (Typer) & Developer UX

Commands

```
$ model-zoo init <model-family> --target gke|k3d
$ model-zoo deploy <model-name> \
      --weights gs://… --gpu a100 --target gke|k3d
$ model-zoo logs <model-name> --tail
$ model-zoo infer <model-name> --file img.jpg --question "…"
```

Deploy flow

1. Build & push image (skip if digest found).
2. Render YAML templates (simple Jinja or `str.format`) to
   `dist/<model>/`.
3. `kubectl apply -f dist/<model>/`.
4. Wait for Service IP (`kubectl get svc -o json`).
5. Run Ray-Client canary:

   ```python
   ray.init("ray://<ip>:10001"); actor = ray.get_actor("<model>");
   ray.get(actor.infer.remote(TEST_PAYLOAD))
   ```
6. On failure → `kubectl delete -f dist/<model>` and exit 1.

Unit tests use `pytest-subprocess` to stub `kubectl`.

Return **“DONE PHASE 4”.**

---

## PHASE 5 – CI/CD & Autoscaling

* GitHub Actions matrix: lint / unit / build / k3d smoke E2E.
* Manual `approve-prod` step.
* Generate `infra/k8s/templates/hpa.yaml` populated by CLI (metric =
  `ray_actor_queue_size`).
* Docs: `docs/production_runbook.md`.
* Grafana dashboard stub `observability/grafana_model_zoo.json`.

Return **“DONE PHASE 5”.**

---

## PHASE 6 – Smoke Tests & Release

* CI job spins up k3d, runs `model-zoo deploy clip … --target k3d`.
* Runs `model-zoo infer clip …` and checks output.
* Update `README.md` (quick-start, add-new-model in ≤ 5 steps).
* Tag repo `v0.1.0`.

Return **“DONE PHASE 6 – PROJECT COMPLETE”.**

---

### Acceptance gates

1. `verify_phase_<N>.sh` pass on macOS & Linux.
2. k3d demo ≤ 12 GB RAM / 5 GB disk.
3. All YAML `kubectl apply --dry-run=client` clean.
4. Synthetic canaries run end-to-end ≤ 180 s on laptop CPU.
5. Docs clearly list how to add a new model in ≤ 5 steps.

---

**End of prompt. Copy above, paste into your coding agent, and start with PHASE 0.**
