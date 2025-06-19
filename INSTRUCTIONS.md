> I’m trying to make a model zoo for some machine learning models. I want to run some CLIP, Grounding DINO + SAM2, and Qwen 2.5 VL for some visual question answering; implementations in Huggingface are fine. I want to use Ray to deploy arbitrary cutting edge ML research models to Kubernetes. I’m going to want to specify something like what GPU to use with nodeSelectors and annotations but otherwise I’m fine with defaults. We will be storing fine-tunes and we have k8s set up so that with the correct metadata annotation for GCP service account and with the right service account name we can automatically access a particular GCS bucket where the fine tunes will live. We want a command line interface for a team to use to each be able to develop and deploy models independently. We will run in GCP on Kubernetes and locally (via minikube/k3s/k3d/whatever makes sense).

Below is a **single, self-contained multi-phase “super-prompt”** you can paste straight into a capable coding-agent session.
It folds together everything we discussed:  *k3d + GKE, one-GPU actors, GCS-pulled weights, cluster-per-model, Ray **Client** ingress (no Ray Serve, no HTTP gateway), autoscaling on queue depth, synthetic canaries, CI/CD, CLI, and docs.*

> **Copy from the next line all the way to the final line and feed it to the agent.**

---

## PHASE 0 – Bootstrap & Contract

```
You are an autonomous coding agent.  
Your mission is to create a production-grade “model-zoo” repo that deploys
Hugging Face CLIP, Grounding DINO + SAM2, and Qwen 2.5 VL models on Ray **(raw)**
running on Kubernetes.  Inference is reached exclusively through the
**Ray Client** protocol (gRPC on :10001); there is **no HTTP gateway**.

### Hard constraints (MUST):

1. Local Kubernetes = **k3d**; cloud = **GKE** (standard or Autopilot).
2. GPU nodes are modern NVIDIA (T4/A100/L4, some 80 GB).  
   *One Ray actor ⇢ one GPU*; horizontal scaling only.
3. Each deploy spins up its **own Ray cluster** (head + workers) via Helm.
4. All weights live in one GCS bucket, hierarchical layout  
   `gs://<bucket>/<project>/<model>/<variant>/<sha>/…` (DVC-style hash dir).  
   Ray workers **always** pull weights at start via GCSFuse.
5. Ray **autoscaler/HPA** keeps `ray_actor_queue_size{name="<model>"} ≤ 1`; scale-to-zero allowed.
6. Every cluster’s head pod exposes:
   * port **10001** – Ray Client  
   * port **8265** – Ray dashboard (optional)
   A Kubernetes **LoadBalancer** (or ClusterIP for k3d) fronts port 10001.
7. Logging: everything to **stdout**; metrics: default Ray + custom latency histogram.
8. CI = **GitHub Actions**; images go to **GCP Artifact Registry**; manual “promote” gate.
9. Auth / network security is handled by VPN & firewall; restrict Service
   `loadBalancerSourceRanges` accordingly.
10. Secrets: **GCP Workload Identity** only (no secrets in repo).
11. CLI must build/push, render Helm, deploy, wait for Service, run a Ray-Client
    canary inference, and **fail hard** (with Helm rollback) if the canary errors.
12. Each usage branch can pin any Ray minor version; no rollback—*roll-forward* only.

### Assumptions (MAY):

- No PII / HIPAA data; no license enforcement.
- Fine-tune retraining jobs live in separate clusters (out of scope here).

### Phase deliverables:

At the end of *every* phase you MUST:
1. Commit all new code / charts / docs to the repo.
2. Update a checklist file `CHECKS.md`.
3. Provide a script `scripts/verify_phase_<N>.sh` that returns exit 0 on success.
4. Ensure all unit tests (`pytest -q`) pass.

Respond **“ACK PHASE 0”** after parsing this contract.
```

---

## PHASE 1 – Repo Skeleton & CI Stub

> **Goal**: directory tree, basic tooling, and a green CI.

Checklist

* `model_zoo/` Python package with `__init__.py`.
* `infra/helm/` chart scaffold.
* `infra/docker/Dockerfile.base-pygpu` (CUDA 12, Python 3.11, uv/poetry).
* `scripts/dev_cluster.sh` (k3d with `--runtime=nvidia` or gpu-strategy).
* `ci.yml` (ruff + pytest cache).
* `verify_phase_1.sh`.

Return **“DONE PHASE 1”** with a brief diffstat.

---

## PHASE 2 – Helm Chart & Ray Actor Template

> **Goal**: produce a parameterised Helm chart and a Python template that spin up a Ray cluster, start the model actor, and expose Ray Client.

### Kubernetes bits

```
Deployment: head pod
  - runs: ray start --head --num-gpus=0 --dashboard-host=0.0.0.0
  - env: MODEL_NAME, WEIGHTS_URI, GPU_TYPE
  - side-car (same image) launches driver.py to create the actor
Service: LoadBalancer (or ClusterIP for k3d)
  - port 10001 (name: ray-client)
HPA: targets Prometheus metric ray_actor_queue_size{name=MODEL_NAME} <= 1
Worker DaemonSet or Deployment:
  - 1 GPU per pod, tolerates GPU nodes, same image
```

### Python bits

```
model_zoo/actors/base.py:
  @ray.remote(num_gpus=1)
  class HFModelActor:
      async def infer(self, payload: dict) -> dict: ...
      async def ready(self) -> bool:  # returns True when weights loaded
model_zoo/driver.py:
  ray.init(address="auto")
  from model_zoo.actors import factory
  actor = factory.make().options(
      name=os.getenv("MODEL_NAME"),
      num_gpus=1).remote(...)
  ray.get(actor.ready.remote())
  ray.get(actor.__ray_terminate__.remote())  # keep actor alive
```

Add `verify_phase_2.sh` that spins up k3d + fake GPU, renders the chart, and checks `helm lint`.

Return **“DONE PHASE 2”**.

---

## PHASE 3 – Canonical Model Actors & Synthetic Canaries

Implement:

1. **CLIPActor** – returns `{similarity: float}` for text + image.
2. **GroundingDINO\_SAM2\_Actor** – returns `{mask_png_b64: str}`.
3. **QwenVLActor** – returns `{answer: str}`.

Fixtures: ≤ 5 tiny inputs per model under `tests/fixtures/`.
`tests/perf/test_canaries.py` starts a local Ray head, launches the actor, asserts:

* `infer()` completes,
* latency < 5 s on CPU,
* no exceptions.

Return **“DONE PHASE 3”.**

---

## PHASE 4 – CLI (Typer) & Developer Workflow

Commands

```
$ model-zoo init <model-family> --target gke|k3d
$ model-zoo deploy <model-name> --weights <gcs://…> --gpu a100 --target gke|k3d
$ model-zoo logs <model-name> --tail
$ model-zoo infer <model-name> --file img.jpg --question "…"
```

Deploy flow

1. Build image (skip if digest exists).
2. Push to Artifact Registry.
3. Render Helm chart to `./dist/<model>`; `helm upgrade --install`.
4. Poll Service until Ray Client port is reachable.
5. Run **canary** (`ray.init("ray://<ip>:10001")`, call `infer()` on test payload).
6. If failure → `helm uninstall` + exit 1.

Return **“DONE PHASE 4”.**

---

## PHASE 5 – CI/CD Hardening & Autoscaling

* Extend GitHub Actions matrix: lint / unit / build / k3d smoke E2E.
* Add `approve-prod` manual job.
* Add `infra/helm/autoscale.yaml` and wire HPA to Prometheus metric.
* `docs/production_runbook.md`.
* `observability/grafana_model_zoo.json` (dashboard stub).

Return **“DONE PHASE 5”.**

---

## PHASE 6 – Smoke Tests & Release

* CI job spins up k3d + fake-GPU, runs `model-zoo deploy clip … --target k3d`.
* Runs `model-zoo infer clip …` and checks output.
* Update `README.md` (3-minute quick-start, adding new model in ≤ 5 steps).
* Tag repo `v0.1.0`.

Return **“DONE PHASE 6 – PROJECT COMPLETE”.**

---

### Acceptance gates for human reviewers

1. `scripts/verify_phase_<N>.sh` pass on macOS & Linux.
2. k3d demo ≤ 12 GB RAM / 5 GB disk.
3. `helm template` has no deprecated `apiVersion`.
4. Synthetic canaries run end-to-end in ≤ 180 s on laptop CPU.
5. Docs clearly list **how to add a new model** in ≤ 5 steps.

---

**End of prompt. Copy above, paste into your coding agent, and start with PHASE 0.**
