"""Command-line interface for model zoo."""
import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import os
import subprocess
import json
import time
from typing import Optional
import ray

app = typer.Typer(help="Model Zoo CLI for deploying ML models on Ray/Kubernetes")
console = Console()


@app.command()
def init(
    model_family: str = typer.Argument(..., help="Model family (clip, grounding-sam2, qwen-vl)"),
    target: str = typer.Option("k3d", help="Deployment target (gke or k3d)")
):
    """Initialize a model family deployment."""
    console.print(f"[green]Initializing {model_family} for {target}...[/green]")
    
    # Create necessary directories
    dist_dir = Path(f"dist/{model_family}")
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Create service account YAML if on GKE
    if target == "gke":
        sa_yaml = f"""apiVersion: v1
kind: ServiceAccount
metadata:
  name: {model_family}-sa
  namespace: default
  annotations:
    iam.gke.io/gcp-service-account: {model_family}@${{GCP_PROJECT}}.iam.gserviceaccount.com
"""
        (dist_dir / "serviceaccount.yaml").write_text(sa_yaml)
    
    console.print(f"[green]✓[/green] Initialized {model_family} in {dist_dir}")


@app.command()
def deploy(
    model_name: str = typer.Argument(..., help="Model name to deploy"),
    weights: str = typer.Option(..., "--weights", help="GCS URI to model weights"),
    gpu: str = typer.Option("nvidia-tesla-t4", "--gpu", help="GPU type"),
    target: str = typer.Option("k3d", "--target", help="Deployment target (gke or k3d)"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Kubernetes namespace"),
    registry: str = typer.Option("gcr.io/myproject", "--registry", help="Container registry")
):
    """Deploy a model to Kubernetes."""
    console.print(f"[bold]Deploying {model_name}...[/bold]")
    
    # Build and push image
    image = f"{registry}/model-zoo-{model_name}:latest"
    console.print(f"[yellow]Building image: {image}[/yellow]")
    
    # Build Docker image
    dockerfile = Path("infra/docker/Dockerfile.model")
    if not dockerfile.exists():
        # Create model Dockerfile
        dockerfile.write_text(f"""FROM model-zoo-base:latest
WORKDIR /app
COPY . .
RUN pip install -e .
""")
    
    build_cmd = ["docker", "build", "-t", image, "-f", str(dockerfile), "."]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Build failed: {result.stderr}[/red]")
        raise typer.Exit(1)
    
    # Push image (skip for k3d)
    if target == "gke":
        console.print(f"[yellow]Pushing image...[/yellow]")
        push_cmd = ["docker", "push", image]
        result = subprocess.run(push_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]Push failed: {result.stderr}[/red]")
            raise typer.Exit(1)
    
    # Render YAML templates
    console.print("[yellow]Rendering YAML templates...[/yellow]")
    dist_dir = Path(f"dist/{model_name}")
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    template_vars = {
        "{{MODEL_NAME}}": model_name,
        "{{IMAGE}}": image,
        "{{WEIGHTS_URI}}": weights,
        "{{GPU_TYPE}}": gpu,
        "{{TARGET_NS}}": namespace,
        "{{APP_LABEL}}": "model-zoo",
        "{{GCP_PROJECT_ID}}": os.environ.get("GCP_PROJECT_ID", "my-project")
    }
    
    # Render each template
    template_dir = Path("infra/k8s/templates")
    for template_file in template_dir.glob("*.yaml"):
        content = template_file.read_text()
        for old, new in template_vars.items():
            content = content.replace(old, new)
        
        output_file = dist_dir / template_file.name
        output_file.write_text(content)
    
    # Apply to Kubernetes
    console.print("[yellow]Applying to Kubernetes...[/yellow]")
    apply_cmd = ["kubectl", "apply", "-f", str(dist_dir), "-n", namespace]
    result = subprocess.run(apply_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Apply failed: {result.stderr}[/red]")
        raise typer.Exit(1)
    
    # Wait for service
    console.print("[yellow]Waiting for service...[/yellow]")
    service_name = f"{model_name}-ray-head"
    
    with console.status("[bold green]Waiting for service to be ready..."):
        for i in range(60):  # Wait up to 5 minutes
            get_svc_cmd = [
                "kubectl", "get", "svc", service_name, 
                "-n", namespace, "-o", "json"
            ]
            result = subprocess.run(get_svc_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                svc = json.loads(result.stdout)
                if target == "k3d":
                    # For k3d, service is immediately available
                    ip = "localhost"
                    break
                else:
                    # For GKE, wait for LoadBalancer IP
                    ingress = svc.get("status", {}).get("loadBalancer", {}).get("ingress", [])
                    if ingress and ingress[0].get("ip"):
                        ip = ingress[0]["ip"]
                        break
            time.sleep(5)
        else:
            console.print("[red]Service failed to become ready[/red]")
            raise typer.Exit(1)
    
    # Run canary test
    console.print(f"[yellow]Running canary test on {ip}:10001...[/yellow]")
    
    try:
        ray.init(f"ray://{ip}:10001")
        actor = ray.get_actor(model_name)
        
        # Simple canary payload
        if "clip" in model_name:
            from tests.fixtures import TEST_IMAGE_B64
            result = ray.get(actor.infer.remote({
                "image_b64": TEST_IMAGE_B64,
                "text": "test"
            }))
            assert "similarity" in result
        
        ray.shutdown()
        console.print("[green]✓ Canary test passed![/green]")
        
    except Exception as e:
        console.print(f"[red]Canary test failed: {e}[/red]")
        console.print("[yellow]Rolling back deployment...[/yellow]")
        delete_cmd = ["kubectl", "delete", "-f", str(dist_dir), "-n", namespace]
        subprocess.run(delete_cmd, capture_output=True)
        raise typer.Exit(1)
    
    # Success
    console.print(f"[green]✓ Model {model_name} deployed successfully![/green]")
    console.print(f"[blue]Ray Client endpoint: ray://{ip}:10001[/blue]")
    console.print(f"[blue]Ray Dashboard: http://{ip}:8265[/blue]")


@app.command()
def logs(
    model_name: str = typer.Argument(..., help="Model name"),
    tail: bool = typer.Option(False, "--tail", "-f", help="Follow log output"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Kubernetes namespace")
):
    """View logs for a deployed model."""
    # Get driver logs
    pod_selector = f"app=model-zoo,model={model_name},component=ray-head"
    
    cmd = [
        "kubectl", "logs", "-n", namespace,
        "-l", pod_selector, "-c", "driver"
    ]
    
    if tail:
        cmd.append("-f")
    
    subprocess.run(cmd)


@app.command()
def infer(
    model_name: str = typer.Argument(..., help="Model name"),
    file: Optional[Path] = typer.Option(None, "--file", help="Input file path"),
    question: Optional[str] = typer.Option(None, "--question", help="Question for VQA models"),
    text: Optional[str] = typer.Option(None, "--text", help="Text for CLIP models"),
    text_prompt: Optional[str] = typer.Option(None, "--text-prompt", help="Text prompt for Grounding DINO"),
    number: Optional[int] = typer.Option(None, "--number", help="Number for is-odd demo model"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Kubernetes namespace")
):
    """Run inference on a deployed model."""
    import base64
    from PIL import Image
    
    # Get service endpoint
    get_svc_cmd = [
        "kubectl", "get", "svc", f"{model_name}-ray-head",
        "-n", namespace, "-o", "json"
    ]
    result = subprocess.run(get_svc_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Model {model_name} not found[/red]")
        raise typer.Exit(1)
    
    svc = json.loads(result.stdout)
    
    # Determine endpoint
    if svc["spec"]["type"] == "LoadBalancer":
        ingress = svc.get("status", {}).get("loadBalancer", {}).get("ingress", [])
        if not ingress or not ingress[0].get("ip"):
            console.print("[red]Service not ready (no external IP)[/red]")
            raise typer.Exit(1)
        ip = ingress[0]["ip"]
    else:
        ip = "localhost"  # k3d
    
    # Build payload based on model type
    if "is-odd" in model_name or "is_odd" in model_name:
        if number is None:
            console.print("[red]--number required for is-odd demo model[/red]")
            raise typer.Exit(1)
        payload = {"number": number}
    else:
        # Load and encode image for image-based models
        if not file:
            console.print(f"[red]--file required for image-based models[/red]")
            raise typer.Exit(1)
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        
        with open(file, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    
        # Build payload for image-based models
        if "clip" in model_name:
            if not text:
                console.print("[red]--text required for CLIP models[/red]")
                raise typer.Exit(1)
            payload = {"image_b64": image_b64, "text": text}
        elif "grounding" in model_name or "sam2" in model_name:
            if not text_prompt:
                console.print("[red]--text-prompt required for Grounding DINO models[/red]")
                raise typer.Exit(1)
            payload = {"image_b64": image_b64, "text_prompt": text_prompt}
        elif "qwen" in model_name:
            if not question:
                console.print("[red]--question required for Qwen VL models[/red]")
                raise typer.Exit(1)
            payload = {"image_b64": image_b64, "question": question}
        else:
            console.print(f"[red]Unknown model type: {model_name}[/red]")
            raise typer.Exit(1)
    
    # Run inference
    console.print(f"[yellow]Running inference on {model_name}...[/yellow]")
    
    try:
        ray.init(f"ray://{ip}:10001")
        actor = ray.get_actor(model_name)
        
        start_time = time.time()
        result = ray.get(actor.infer.remote(payload))
        inference_time = time.time() - start_time
        
        ray.shutdown()
        
        # Display results
        console.print(f"[green]✓ Inference completed in {inference_time:.2f}s[/green]")
        
        table = Table(title="Results")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in result.items():
            if key == "mask_png_b64":
                # Save mask to file
                mask_file = Path(f"{model_name}_mask.png")
                with open(mask_file, "wb") as f:
                    f.write(base64.b64decode(value))
                table.add_row(key, f"Saved to {mask_file}")
            else:
                table.add_row(key, str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Inference failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list(namespace: str = typer.Option("default", "--namespace", "-n", help="Kubernetes namespace")):
    """List deployed models."""
    cmd = [
        "kubectl", "get", "deployments",
        "-n", namespace,
        "-l", "app=model-zoo",
        "-o", "json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Failed to list models: {result.stderr}[/red]")
        raise typer.Exit(1)
    
    data = json.loads(result.stdout)
    
    table = Table(title="Deployed Models")
    table.add_column("Model", style="cyan")
    table.add_column("Ready", style="green")
    table.add_column("Image", style="yellow")
    
    for item in data.get("items", []):
        name = item["metadata"]["labels"].get("model", "unknown")
        ready = f"{item['status'].get('readyReplicas', 0)}/{item['spec']['replicas']}"
        image = item["spec"]["template"]["spec"]["containers"][0]["image"]
        
        table.add_row(name, ready, image)
    
    console.print(table)


@app.command()
def delete(
    model_name: str = typer.Argument(..., help="Model name to delete"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Kubernetes namespace"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """Delete a deployed model."""
    if not yes:
        confirm = typer.confirm(f"Delete model {model_name}?")
        if not confirm:
            raise typer.Abort()
    
    dist_dir = Path(f"dist/{model_name}")
    if not dist_dir.exists():
        console.print(f"[red]No deployment files found for {model_name}[/red]")
        raise typer.Exit(1)
    
    cmd = ["kubectl", "delete", "-f", str(dist_dir), "-n", namespace]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        console.print(f"[red]Delete failed: {result.stderr}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓ Model {model_name} deleted[/green]")


if __name__ == "__main__":
    app()