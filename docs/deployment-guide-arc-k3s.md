# Deployment Guide — Arc-enabled K3s + Azure IoT Operations

Step-by-step record of deploying the MOMENT inference server to an
Arc-enabled K3s cluster running Azure IoT Operations (AIO).

---

## Cluster Details

| Property | Value |
|----------|-------|
| Cluster name | `time-series-ml` |
| Type | Arc-enabled K3s (NOT AKS) |
| Nodes | 3 (master0 + 2 agents) |
| K3s version | v1.33.6+k3s1 |
| AIO instance | `aio-svwap` |
| Resource group | `time-series-ml-161301359` |
| Subscription | `650161eb-ba54-4616-9688-60d115887fec` |
| ACR | `aioedgeintel.azurecr.io` |

---

## Prerequisites

1. **Azure CLI** with `connectedk8s` extension
2. **kubectl** configured for the cluster
3. **Helm** v3
4. Arc proxy access (for remote K3s clusters)

## Step 1: Connect to the Cluster

For Arc-enabled clusters (not AKS), use the connectedk8s proxy:

```bash
az connectedk8s proxy \
  --name time-series-ml \
  --resource-group time-series-ml-161301359 \
  --subscription 650161eb-ba54-4616-9688-60d115887fec \
  --port 47011
```

In another terminal, verify:

```bash
kubectl --context time-series-ml get nodes
kubectl --context time-series-ml get pods -n azure-iot-operations
```

## Step 2: Create ACR and Push Images

```bash
# Create ACR (if needed)
az acr create -n aioedgeintel \
  -g time-series-ml-161301359 \
  --sku Basic

# Enable admin (required for K3s pull secret)
az acr update -n aioedgeintel --admin-enabled true

# Build images via ACR Tasks (no local Docker needed)
az acr build --registry aioedgeintel \
  --image inference-server:0.1.8 \
  --build-arg BUILD_TARGET=cpu \
  --file src/inference-server/Dockerfile \
  src/inference-server/

az acr build --registry aioedgeintel \
  --image sensor-simulator:0.1.3 \
  --file src/sensor-simulator/Dockerfile \
  src/sensor-simulator/
```

## Step 3: Create Image Pull Secret

Since this is K3s (not AKS), you can't use `az aks update --attach-acr`.
Instead, create a pull secret:

```bash
ACR_USER=$(az acr credential show -n aioedgeintel --query username -o tsv)
ACR_PASS=$(az acr credential show -n aioedgeintel --query "passwords[0].value" -o tsv)

kubectl --context time-series-ml create secret docker-registry acr-secret \
  --namespace azure-iot-operations \
  --docker-server=aioedgeintel.azurecr.io \
  --docker-username="$ACR_USER" \
  --docker-password="$ACR_PASS"
```

## Step 4: Deploy via Helm

```bash
helm install aio-si ./deploy/aio-connected/helm/aio-sensor-intelligence \
  --namespace azure-iot-operations \
  --kube-context time-series-ml \
  -f deploy/aio-connected/helm/values-time-series-ml.yaml
```

For upgrades:

```bash
helm upgrade aio-si ./deploy/aio-connected/helm/aio-sensor-intelligence \
  --namespace azure-iot-operations \
  --kube-context time-series-ml \
  -f deploy/aio-connected/helm/values-time-series-ml.yaml \
  --wait --timeout 5m
```

## Step 5: Verify Deployment

```bash
# Check pods
kubectl --context time-series-ml get pods -n azure-iot-operations | grep aio-si

# Check inference server logs (model load takes ~12s)
kubectl --context time-series-ml logs -l component=inference-server \
  -n azure-iot-operations --tail=20

# Port-forward and test
kubectl --context time-series-ml port-forward \
  svc/aio-si-aio-sensor-intelligence-inference-server 8080:8080 \
  -n azure-iot-operations

curl http://localhost:8080/health
# Expected: {"status":"healthy","model":"moment","ready":true}
```

## Step 6: Run API Tests

```bash
python samples/04-http-api-tests/test_moment_api.py
# Expected: 8/8 tests pass
```

---

## Issues Encountered & Resolutions

### 1. momentfm Version

**Problem**: `pyproject.toml` specified `momentfm>=1.0.0` but only `0.1.x` versions
exist on PyPI.

**Fix**: Changed to `momentfm>=0.1.4`.

### 2. numpy Conflict

**Problem**: `momentfm==0.1.4` pins `numpy==1.25.2` which conflicts with other deps
requiring `numpy>=1.26.0`.

**Fix**: Install momentfm with `--no-deps` in Dockerfile, then install numpy and
transformers separately.

### 3. AIO Broker Authentication

**Problem**: AIO MQTT broker requires SAT (ServiceAccountToken) authentication with
audience `aio-internal`. The original code didn't support this.

**Fix**: Added SAT auth support to both inference server and simulator MQTT clients.
Created ServiceAccount in Helm chart with projected token volume.

### 4. MOMENT forward() API

**Problem**: MOMENT uses keyword-only arguments (`x_enc=`, `input_mask=`). The code
was passing the tensor as a positional argument.

**Fix**: Updated `moment_provider.py` to use keyword args and provide `input_mask`.

### 5. MOMENT task_name

**Problem**: The pretrained model always loads with `task_name='reconstruction'`
regardless of what's passed to `from_pretrained()`.

**Fix**: Call task-specific methods directly (`pipeline.reconstruction()`,
`pipeline.forecast()`) instead of relying on `forward()` routing.

### 6. Relative Imports

**Problem**: `tasks/` module used `from ..model_providers` relative imports which fail
when uvicorn runs from `/app` directory.

**Fix**: Changed to absolute imports (`from model_providers.base import ...`).

### 7. Port-Forward Instability

**Problem**: `kubectl port-forward` through Arc proxy drops connections frequently.

**Workaround**: Use async port-forward sessions and restart as needed. For production,
use cluster-internal routing.

---

## Resource Requirements

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 500m | 2 cores |
| Memory | 2Gi | 4Gi |
| GPU | — | — (CPU only) |
| Disk | ~1.5 GB (model cache) | — |

The MOMENT-1-large model needs ~2.5 GB RAM at inference time. The 4Gi limit
provides headroom for spikes.
