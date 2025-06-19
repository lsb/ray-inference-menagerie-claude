# Model Zoo Production Runbook

## Overview

This runbook covers operational procedures for the Model Zoo production environment running on Google Kubernetes Engine (GKE).

## Architecture

- **Ray Clusters**: Each model deployment creates its own Ray cluster (head + workers)
- **GPU Allocation**: One Ray actor per GPU, horizontal scaling only
- **Storage**: Model weights stored in GCS with hierarchical layout
- **Networking**: Ray Client protocol on port 10001, no HTTP gateway
- **Autoscaling**: Based on `ray_actor_queue_size` metric with scale-to-zero support

## Prerequisites

### Required Tools
```bash
gcloud auth login
kubectl config current-context  # Should point to production cluster
```

### Environment Variables
```bash
export GCP_PROJECT_ID="your-project-id"
export GKE_CLUSTER="model-zoo-prod"
export GKE_ZONE="us-central1-a"
```

## Deployment Procedures

### 1. Standard Model Deployment

```bash
# Deploy a new model
model-zoo deploy clip-vit-large \
  --weights gs://model-zoo-weights/clip/vit-large/v1.0/weights \
  --gpu nvidia-tesla-a100 \
  --target gke \
  --namespace production

# Verify deployment
model-zoo list --namespace production
```

### 2. Emergency Rollback

```bash
# If canary test fails, automatic rollback occurs
# For manual rollback:
model-zoo delete <model-name> --namespace production --yes

# Check for stuck resources
kubectl get pods -n production -l app=model-zoo,model=<model-name>
kubectl delete pod -n production -l app=model-zoo,model=<model-name> --force
```

### 3. Scaling Operations

```bash
# Check current scaling status
kubectl get hpa -n production

# Manually scale if needed
kubectl scale deployment <model-name>-ray-worker --replicas=5 -n production

# View autoscaling events
kubectl describe hpa <model-name>-ray-worker-hpa -n production
```

## Monitoring & Observability

### Key Metrics to Monitor

1. **Ray Actor Queue Size**: `ray_actor_queue_size{model_name="<model>"}`
2. **GPU Utilization**: `nvidia_gpu_utilization`
3. **Inference Latency**: `ray_serve_request_latency_ms`
4. **Pod Ready Status**: `kube_pod_status_ready`
5. **HPA Events**: `kube_hpa_status_current_replicas`

### Dashboard Access

- **Ray Dashboard**: `kubectl port-forward svc/<model-name>-ray-head 8265:8265 -n production`
- **Grafana**: [Internal Grafana URL]
- **GCP Monitoring**: [GCP Console URL]

### Log Access

```bash
# Driver logs
model-zoo logs <model-name> --namespace production

# Worker logs
kubectl logs -l app=model-zoo,component=ray-worker,model=<model-name> -n production

# Real-time logs
kubectl logs -f deployment/<model-name>-ray-head -c driver -n production
```

## Troubleshooting

### Common Issues

#### 1. Model Not Ready

**Symptoms**: `model-zoo infer` fails with connection timeout

**Diagnosis**:
```bash
kubectl get pods -n production -l model=<model-name>
kubectl describe pod <pod-name> -n production
```

**Resolution**:
- Check if model weights are accessible in GCS
- Verify GPU resources are available
- Check Ray head node status

#### 2. Autoscaling Not Working

**Symptoms**: No workers scaling despite high queue size

**Diagnosis**:
```bash
kubectl describe hpa <model-name>-ray-worker-hpa -n production
kubectl get --raw /apis/external.metrics.k8s.io/v1beta1/namespaces/production/ray_actor_queue_size
```

**Resolution**:
- Verify metrics server is running
- Check if custom metrics adapter is configured
- Ensure HPA has correct metric selectors

#### 3. High Inference Latency

**Symptoms**: Requests taking > 5 seconds

**Diagnosis**:
```bash
# Check GPU utilization
kubectl exec -it <worker-pod> -n production -- nvidia-smi

# Check Ray dashboard
kubectl port-forward svc/<model-name>-ray-head 8265:8265 -n production
```

**Resolution**:
- Scale up workers if queue is backing up
- Check for GPU memory issues
- Verify model is loaded correctly

#### 4. Out of GPU Resources

**Symptoms**: Workers stuck in Pending state

**Diagnosis**:
```bash
kubectl describe nodes | grep -A 10 "nvidia.com/gpu"
kubectl get pods -A | grep Pending
```

**Resolution**:
- Add more GPU nodes to cluster
- Remove unused model deployments
- Check node selectors and taints

### Emergency Procedures

#### Complete Service Outage

1. **Assess Impact**:
   ```bash
   kubectl get deployments -n production -l app=model-zoo
   kubectl get services -n production -l app=model-zoo
   ```

2. **Check Cluster Health**:
   ```bash
   kubectl get nodes
   kubectl top nodes
   ```

3. **Restore Service**:
   ```bash
   # Restart all model deployments
   kubectl rollout restart deployment -n production -l app=model-zoo
   
   # Or redeploy from scratch
   for model in $(kubectl get deployments -n production -l app=model-zoo -o name); do
     kubectl delete $model -n production
   done
   ```

#### Resource Exhaustion

1. **Immediate Actions**:
   ```bash
   # Scale down non-critical models
   kubectl scale deployment <model-name>-ray-worker --replicas=0 -n production
   
   # Delete completed/failed pods
   kubectl delete pods --field-selector=status.phase=Succeeded -n production
   kubectl delete pods --field-selector=status.phase=Failed -n production
   ```

2. **Add Resources**:
   ```bash
   # Scale up node pool (if using GKE Autopilot, this is automatic)
   gcloud container clusters resize $GKE_CLUSTER --num-nodes=10 --zone=$GKE_ZONE
   ```

## Maintenance Windows

### Weekly Maintenance (Sundays 2-4 AM PST)

1. **Update base images** (if new security patches)
2. **Clean up unused resources**:
   ```bash
   kubectl delete pods --field-selector=status.phase=Succeeded -n production
   docker system prune -f
   ```
3. **Review and rotate logs**
4. **Check certificate expiration**

### Monthly Maintenance

1. **Update Kubernetes cluster**
2. **Review autoscaling policies**
3. **Analyze cost optimization opportunities**
4. **Update security configurations**

## Security Considerations

### Access Control

- All model deployments use Workload Identity
- No secrets stored in repository
- LoadBalancer services restricted to VPN CIDR ranges
- RBAC policies limit namespace access

### Monitoring

- All deployments and access are logged
- Anomaly detection on inference patterns
- Regular security scans of container images

## Performance Optimization

### Cost Management

- Use preemptible nodes for development workloads
- Implement scale-to-zero for unused models
- Monitor GPU utilization and right-size instances

### Latency Optimization

- Co-locate models with similar traffic patterns
- Use local SSD for model caching
- Optimize batch sizes for GPU utilization

## Contact Information

- **On-call Engineer**: [Slack: #model-zoo-oncall]
- **Platform Team**: [platform-team@company.com]
- **Escalation**: [engineering-leads@company.com]

## Runbook Version

Version: 1.0  
Last Updated: 2025-06-19  
Next Review: 2025-07-19