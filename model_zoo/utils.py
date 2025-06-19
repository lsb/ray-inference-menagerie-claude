"""Utility functions for model zoo."""
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional


def run_command(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def kubectl_get(resource: str, name: Optional[str] = None, namespace: str = "default") -> Dict[str, Any]:
    """Get a Kubernetes resource as JSON."""
    cmd = ["kubectl", "get", resource]
    if name:
        cmd.append(name)
    cmd.extend(["-n", namespace, "-o", "json"])
    
    result = run_command(cmd)
    return json.loads(result.stdout)


def kubectl_apply(yaml_dir: Path, namespace: str = "default") -> bool:
    """Apply YAML files to Kubernetes."""
    cmd = ["kubectl", "apply", "-f", str(yaml_dir), "-n", namespace]
    result = run_command(cmd, check=False)
    return result.returncode == 0


def kubectl_delete(yaml_dir: Path, namespace: str = "default") -> bool:
    """Delete resources from YAML files."""
    cmd = ["kubectl", "delete", "-f", str(yaml_dir), "-n", namespace]
    result = run_command(cmd, check=False)
    return result.returncode == 0


def docker_build(image: str, dockerfile: Path, context: Path = Path(".")) -> bool:
    """Build a Docker image."""
    cmd = ["docker", "build", "-t", image, "-f", str(dockerfile), str(context)]
    result = run_command(cmd, check=False)
    return result.returncode == 0


def docker_push(image: str) -> bool:
    """Push a Docker image."""
    cmd = ["docker", "push", image]
    result = run_command(cmd, check=False)
    return result.returncode == 0


def wait_for_service(service_name: str, namespace: str = "default", timeout: int = 300) -> Optional[str]:
    """Wait for a LoadBalancer service to get an external IP."""
    import time
    
    for _ in range(timeout // 5):
        try:
            svc = kubectl_get("svc", service_name, namespace)
            if svc["spec"]["type"] == "LoadBalancer":
                ingress = svc.get("status", {}).get("loadBalancer", {}).get("ingress", [])
                if ingress and ingress[0].get("ip"):
                    return ingress[0]["ip"]
            else:
                # ClusterIP or NodePort - assume localhost for k3d
                return "localhost"
        except Exception:
            pass
        time.sleep(5)
    
    return None


def render_template(template_file: Path, variables: Dict[str, str]) -> str:
    """Render a template file with variables."""
    content = template_file.read_text()
    for old, new in variables.items():
        content = content.replace(old, new)
    return content