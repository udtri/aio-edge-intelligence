# Architecture

## System Overview

**aio-sensor-intelligence** brings time-series foundation models (starting
with [MOMENT](https://github.com/moment-timeseries-foundation-model/moment))
to the industrial edge via Azure IoT Operations (AIO).  It runs inference
on raw sensor streams in real time — detecting anomalies, forecasting
trends, and classifying operating regimes — without requiring any
model training or fine-tuning.

```
┌──────────────┐       MQTT        ┌──────────────────┐       MQTT        ┌────────────────┐
│   Sensors /  │  ──────────────►  │   MQTT Broker    │  ──────────────►  │  Inference      │
│   Simulator  │   sensors/{type}  │  (Mosquitto/AIO) │   sensors/{type}  │  Server         │
└──────────────┘   /{device_id}    └──────────────────┘   /{device_id}    │  (FastAPI +     │
                                           │                              │   MOMENT)       │
                                           │                              └───────┬────────┘
                                           │                                      │
                                           │  ai/results                          │
                                           ◄──────────────────────────────────────┘
                                           │
                                   ┌───────▼──────────┐
                                   │  Consumers        │
                                   │  (Dashboards,     │
                                   │   AIO Dataflows,  │
                                   │   Cloud Alerts)   │
                                   └──────────────────┘
```

## Data Flow

1. **Ingest** — Sensors (or the built-in simulator) publish JSON readings
   to MQTT topics under `sensors/{type}/{device_id}`.
2. **Buffer** — The inference server subscribes to these topics, accumulates
   readings in a per-device sliding window, and triggers inference once the
   window reaches the required length (default 512 samples).
3. **Infer** — The MOMENT model performs anomaly detection (reconstruction)
   and/or forecasting on the buffered window.
4. **Publish** — Results are published back to the broker on the
   `ai/results` topic as JSON.
5. **Consume** — Downstream consumers — dashboards, AIO dataflows, or cloud
   services — subscribe to `ai/results`.

---

## Components

### Inference Server (`src/inference/`)

A FastAPI application that:

| Responsibility | Detail |
|---|---|
| MQTT subscription | Listens to configurable sensor topics |
| Sliding-window buffer | Maintains a per-device ring buffer |
| Model inference | Calls the active model provider |
| Result publishing | Publishes JSON results to `ai/results` |
| Health endpoint | `GET /health` for liveness/readiness probes |

Configuration is entirely via environment variables (see the root README).

### Sensor Simulator (`src/simulator/`)

Generates realistic multi-sensor data (vibration, temperature, pressure)
with configurable anomaly injection.  Publishes at a configurable rate
(default 10 Hz) and supports multiple simulated devices.

### MQTT Bridge / Broker

In **standalone mode** the project deploys a Mosquitto container.  In
**AIO-connected mode** the AIO MQTT broker is used directly — no extra
broker is needed.

### Model Provider Abstraction (`src/inference/models/`)

All model interactions go through an abstract `ModelProvider` interface:

```python
class ModelProvider(ABC):
    @abstractmethod
    def detect_anomalies(self, window: np.ndarray) -> AnomalyResult: ...

    @abstractmethod
    def forecast(self, window: np.ndarray, horizon: int) -> ForecastResult: ...
```

Currently implemented:

| Provider | Model | Notes |
|---|---|---|
| `MomentProvider` | MOMENT-1-large | Default; runs on CPU or GPU |

Adding a new model (e.g., TimesFM) requires only a new
`ModelProvider` subclass — zero changes to the rest of the system.

---

## Deployment Modes

### Mode 1 — Standalone (Docker Compose)

Best for **development, demos, and quick evaluation**.

```
docker compose up
```

This starts three containers:

| Container | Image | Purpose |
|---|---|---|
| `mosquitto` | eclipse-mosquitto | MQTT broker |
| `inference` | aio-sensor-intelligence/inference | Model inference |
| `simulator` | aio-sensor-intelligence/simulator | Test data generator |

No Azure subscription, no Kubernetes cluster, no Arc enrollment required.

### Mode 2 — AIO-Connected (Helm)

Best for **production and integration with Azure IoT Operations**.

Prerequisites: a Kubernetes cluster with Azure Arc and Azure IoT
Operations installed (see `docs/prerequisites.md`).

```bash
helm install aio-sensor-intelligence deploy/helm/aio-sensor-intelligence \
    --set aio.mqttBroker=aio-broker-endpoint \
    --namespace azure-iot-operations
```

In this mode:

* The inference server connects to the **AIO MQTT broker** instead of a
  local Mosquitto instance.
* AIO **Dataflows** can be configured to route `ai/results` to Azure
  Event Hubs, Data Explorer, or other cloud services.
* AIO **Akri** can be used for device discovery.

---

## MQTT Topic Schema

### Sensor Data (ingress)

**Topic pattern:** `sensors/{sensor_type}/{device_id}`

Example: `sensors/vibration/motor-pump-01`

```json
{
    "timestamp": "2025-01-15T10:30:00.000Z",
    "sensor_id": "motor-pump-01",
    "vibration": 1.2345,
    "temperature": 55.3,
    "pressure": 101.5
}
```

### Inference Results (egress)

**Topic:** `ai/results`

```json
{
    "timestamp": "2025-01-15T10:30:05.000Z",
    "sensor_id": "motor-pump-01",
    "task": "anomaly_detection",
    "anomaly_score": 0.92,
    "is_anomaly": true,
    "reconstruction_error": [0.01, 0.02, ..., 3.45, ...],
    "model": "MOMENT-1-large",
    "window_size": 512,
    "processing_time_ms": 45
}
```

```json
{
    "timestamp": "2025-01-15T10:30:05.000Z",
    "sensor_id": "motor-pump-01",
    "task": "forecasting",
    "forecast_horizon": 96,
    "forecast_values": [1.23, 1.25, ...],
    "model": "MOMENT-1-large",
    "window_size": 512,
    "processing_time_ms": 52
}
```

---

## Sliding Window Buffering Strategy

The inference server maintains a **per-device ring buffer** for each
subscribed sensor topic.

```
Incoming samples → [ . . . . . . . . . . . . ] → oldest samples discarded
                   ◄──── window_size (512) ────►
```

| Parameter | Default | Description |
|---|---|---|
| `WINDOW_SIZE` | 512 | Number of samples in the inference window |
| `STRIDE` | 64 | Samples between consecutive inferences |
| `MAX_DEVICES` | 100 | Maximum concurrent device buffers |

When the buffer for a device reaches `WINDOW_SIZE` samples, inference
runs.  Subsequent inferences occur every `STRIDE` new samples (i.e.,
the window slides forward).

This means:
- First inference fires after 512 samples (~51 s at 10 Hz).
- Subsequent inferences fire every 64 samples (~6.4 s at 10 Hz).
- Overlapping windows provide continuity for anomaly detection.

---

## AIO Dataflow Integration

When deployed in AIO-connected mode, Azure IoT Operations **Dataflows**
can route inference results to cloud services:

```yaml
apiVersion: connectivity.iotoperations.azure.com/v1
kind: Dataflow
metadata:
  name: anomaly-to-eventhubs
spec:
  operations:
    - operationType: source
      sourceSettings:
        endpointRef: aio-broker
        dataSources:
          - ai/results
    - operationType: filter
      filterSettings:
        expression: "$payload.is_anomaly == true"
    - operationType: destination
      destinationSettings:
        endpointRef: eventhubs-endpoint
        dataDestination: anomaly-alerts
```

This enables cloud-side dashboards, alerting, and long-term storage
without modifying any edge component.
