# Prerequisites

This guide covers what you need for both deployment modes.

---

## Mode 1 — Standalone (Docker Compose)

The standalone mode has minimal requirements and is the fastest way to
get started.

### Required

| Tool | Version | Purpose |
|---|---|---|
| **Docker** | 20.10+ | Container runtime |
| **Docker Compose** | v2+ | Multi-container orchestration |
| **Python** | 3.10+ | Running samples & notebooks |

### Optional

| Tool | Purpose |
|---|---|
| **NVIDIA GPU + CUDA** | Accelerated inference (CPU works fine) |
| **mosquitto-clients** | `mosquitto_pub` / `mosquitto_sub` for debugging |
| **Any Kubernetes cluster** | If you want to deploy with Helm but without AIO |

### Setup Steps

1. **Install Docker Desktop** (or Docker Engine on Linux):
   - Windows / macOS: <https://docs.docker.com/desktop/>
   - Linux: <https://docs.docker.com/engine/install/>

2. **Install Python 3.10+**:
   - <https://www.python.org/downloads/>
   - Or use `pyenv` / `conda` to manage versions.

3. **Clone the repo and start services**:
   ```bash
   git clone https://github.com/<org>/aio-sensor-intelligence.git
   cd aio-sensor-intelligence
   docker compose up -d
   ```

4. **Verify**:
   ```bash
   # Check all containers are running
   docker compose ps

   # Test the inference server health endpoint
   curl http://localhost:5000/health

   # Run the quick test
   cd samples/01-quick-test
   pip install momentfm torch numpy
   python test_moment.py
   ```

---

## Mode 2 — AIO-Connected (Kubernetes + Azure IoT Operations)

This mode integrates with Azure IoT Operations for production edge
deployments.

### Required

| Tool | Version | Purpose |
|---|---|---|
| **Kubernetes cluster** | 1.25+ | Any distribution (see below) |
| **Azure CLI** | 2.50+ | Azure resource management |
| **Azure CLI extensions** | `connectedk8s`, `iot-ops` | Arc and AIO management |
| **Helm** | 3.10+ | Deploying the inference server |
| **kubectl** | 1.25+ | Cluster interaction |

### Supported Kubernetes Distributions

| Distribution | Notes |
|---|---|
| **K3s** | Lightweight; great for single-node edge devices |
| **AKS** | Azure Kubernetes Service (cloud or hybrid) |
| **AKS Edge Essentials** | Designed for Windows IoT edge devices |
| **MicroK8s** | Canonical's lightweight Kubernetes |
| **Kind / Minikube** | Local development (not for production) |

> **Key point:** Azure IoT Operations runs on *any* Arc-enabled
> Kubernetes cluster — you are not locked into a specific distribution.

### Setup Steps

#### 1. Prepare a Kubernetes Cluster

**Option A — K3s (Linux edge device):**
```bash
curl -sfL https://get.k3s.io | sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
```

**Option B — AKS Edge Essentials (Windows):**

Follow the official guide:
<https://learn.microsoft.com/azure/aks/hybrid/aks-edge-overview>

**Option C — AKS (cloud):**
```bash
az aks create -g myResourceGroup -n myCluster --node-count 1
az aks get-credentials -g myResourceGroup -n myCluster
```

#### 2. Arc-Enable the Cluster

```bash
az connectedk8s connect \
    --name my-edge-cluster \
    --resource-group myResourceGroup \
    --location eastus2
```

Verify:
```bash
az connectedk8s show --name my-edge-cluster --resource-group myResourceGroup
```

#### 3. Install Azure IoT Operations

```bash
# Initialize IoT Operations on the cluster
az iot ops init \
    --cluster my-edge-cluster \
    --resource-group myResourceGroup

# Deploy IoT Operations components
az iot ops create \
    --cluster my-edge-cluster \
    --resource-group myResourceGroup \
    --name my-aio-instance
```

For full details see the official documentation:
<https://learn.microsoft.com/azure/iot-operations/deploy-iot-ops/howto-deploy-iot-operations>

#### 4. Deploy aio-sensor-intelligence

```bash
helm install aio-sensor-intelligence deploy/helm/aio-sensor-intelligence \
    --namespace azure-iot-operations \
    --set aio.mqttBroker=aio-broker:18883 \
    --set inference.model=AutonLab/MOMENT-1-large \
    --set inference.device=cpu
```

#### 5. Verify

```bash
# Check pods
kubectl get pods -n azure-iot-operations -l app=aio-sensor-intelligence

# View logs
kubectl logs -n azure-iot-operations -l app=aio-sensor-intelligence -f

# Test health
kubectl port-forward svc/aio-sensor-intelligence 5000:5000 -n azure-iot-operations
curl http://localhost:5000/health
```

---

## Network Requirements

| Direction | Port | Protocol | Purpose |
|---|---|---|---|
| Inbound (edge) | 1883 | MQTT | Sensor data ingestion (standalone) |
| Inbound (edge) | 5000 | HTTP | Inference server health endpoint |
| Outbound (edge → cloud) | 443 | HTTPS | Azure Arc agent, container registry pulls |
| Internal (cluster) | 18883 | MQTT | AIO broker (AIO mode) |

---

## Useful Links

- [Azure IoT Operations documentation](https://learn.microsoft.com/azure/iot-operations/)
- [Azure Arc-enabled Kubernetes](https://learn.microsoft.com/azure/azure-arc/kubernetes/)
- [AKS Edge Essentials](https://learn.microsoft.com/azure/aks/hybrid/aks-edge-overview)
- [K3s documentation](https://docs.k3s.io/)
- [MOMENT model on Hugging Face](https://huggingface.co/AutonLab/MOMENT-1-large)
