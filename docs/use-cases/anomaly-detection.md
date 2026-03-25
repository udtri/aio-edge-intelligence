# Use Case: Real-Time Anomaly Detection in Manufacturing

## Scenario

Modern manufacturing processes — chemical plants, semiconductor fabs,
food processing lines, and assembly operations — generate **thousands of
sensor readings per second** across dozens of process variables.

Operators cannot manually monitor all signals simultaneously.  Subtle
multi-variable deviations that precede quality defects or equipment
failures are easy to miss.

This use case deploys MOMENT as a **real-time process monitor** that
watches all sensor streams, detects anomalous patterns, and raises
alerts — enabling operators to intervene before defects propagate.

---

## Multi-Sensor Fusion Approach

Instead of monitoring each sensor independently, this setup uses
MOMENT's ability to reason about **temporal patterns** to detect
anomalies that only appear when multiple signals are considered together.

### Architecture

```
  Sensor A (temperature)  ──┐
  Sensor B (pressure)     ──┼──►  Sliding Window  ──►  MOMENT  ──►  Anomaly Score
  Sensor C (flow rate)    ──┤      (per process       (reconstruction)
  Sensor D (vibration)    ──┘       unit)
```

Each process unit (reactor, press, conveyor section) gets its own
sliding window.  Sensors are aggregated by **timestamp alignment** —
the inference server matches readings from the same time window across
multiple sensor topics.

### Why Multi-Sensor Matters

| Scenario | Single-Sensor Detection | Multi-Sensor Detection |
|---|---|---|
| Temperature rises 2°C | Normal (within spec) | Anomaly: temperature up while pressure steady — heat exchanger fouling |
| Vibration increases slightly | Normal | Anomaly: vibration + temperature rising together — bearing degradation |
| Flow rate drops 5% | Possibly normal | Anomaly: flow down + pressure up — partial blockage |

---

## Setup

### 1. Configure Sensor Topics

Publish each sensor type to its own MQTT topic:

```
sensors/temperature/{process_unit_id}
sensors/pressure/{process_unit_id}
sensors/flow/{process_unit_id}
sensors/vibration/{process_unit_id}
```

Payload format:
```json
{
    "timestamp": "2025-01-15T10:30:00.000Z",
    "sensor_id": "reactor-01-temp",
    "value": 185.3,
    "unit": "celsius"
}
```

### 2. Configure the Inference Server

```bash
# Subscribe to all sensor types for all process units
MQTT_TOPICS=sensors/+/+

# Inference settings
WINDOW_SIZE=512
STRIDE=32          # More frequent checks for process monitoring
ANOMALY_THRESHOLD=0.6

# Tasks
TASKS=anomaly_detection
```

### 3. Deploy

**Standalone (Docker Compose):**
```bash
docker compose up -d
```

**AIO-connected (Helm):**
```bash
helm install aio-sensor-intelligence deploy/helm/aio-sensor-intelligence \
    --namespace azure-iot-operations \
    --set mqtt.topics="sensors/+/+" \
    --set inference.stride=32 \
    --set inference.anomalyThreshold=0.6
```

---

## Result Payload

```json
{
    "timestamp": "2025-01-15T10:30:05.000Z",
    "sensor_id": "reactor-01",
    "task": "anomaly_detection",
    "anomaly_score": 0.78,
    "is_anomaly": true,
    "reconstruction_error_mean": 0.78,
    "reconstruction_error_max": 2.34,
    "top_anomalous_regions": [
        {"start_idx": 420, "end_idx": 445, "severity": 0.91},
        {"start_idx": 380, "end_idx": 395, "severity": 0.72}
    ],
    "model": "MOMENT-1-large",
    "window_size": 512,
    "processing_time_ms": 38
}
```

---

## Multi-Sensor Fusion Architecture

For complex manufacturing processes, monitoring sensors independently
is not enough.  Correlated deviations across multiple channels — e.g.
temperature rising while pressure drops — often indicate the earliest
stages of a fault.

The **SensorFusionEngine** (in `samples/03-end-to-end/multi_sensor_fusion.py`)
implements a lightweight fusion layer that sits between per-channel
MOMENT inference and the alerting pipeline.

### Architecture Diagram

```
                ┌──────────────┐
  Temp sensor ──┤  MOMENT      ├──► score_temp ──┐
                │  (per-channel │                 │
  Pres sensor ──┤   anomaly    ├──► score_pres ──┼──► SensorFusionEngine ──► fused_score
                │   detection) │                 │         │
  Vib  sensor ──┤              ├──► score_vib  ──┘         ▼
                └──────────────┘                    is_anomaly? ──► MQTT publish
                                                                   ──► AIO Dataflow
```

### Fusion Strategies

| Strategy           | Formula                              | Best for                        |
|--------------------|--------------------------------------|---------------------------------|
| `weighted_average` | Σ(weight × score) / Σ(weight)       | General process monitoring      |
| `max`              | max(score across channels)           | Safety-critical — alert on any  |
| `voting`           | fraction of channels above threshold | Reducing false positives         |

### Code Example

```python
from multi_sensor_fusion import SensorFusionEngine

engine = SensorFusionEngine(strategy="weighted_average", global_threshold=0.7)

# Register channels with individual weights and thresholds
engine.add_channel("temperature", weight=1.2, threshold=0.65)
engine.add_channel("pressure",    weight=1.0, threshold=0.70)
engine.add_channel("vibration",   weight=0.8, threshold=0.60)

# After running MOMENT on each channel's window:
engine.update("temperature", 0.45)
engine.update("pressure",    0.80)
engine.update("vibration",   0.30)

print(engine.fused_score())   # 0.5133 (weighted average)
print(engine.is_anomaly())    # False (below 0.7)

# Get detailed per-channel + fused status
status = engine.get_status()
for ch in status["channels"]:
    print(f"  {ch['name']}: score={ch['latest_score']:.2f}  anomaly={ch['is_anomaly']}")
```

---

## MQTT Topic Mapping for Multi-Sensor Scenarios

When running with the `MQTTBridge`, each sensor type publishes to a
dedicated topic hierarchy.  The inference server subscribes to all of
them via a wildcard and routes results to a parallel results tree.

### Input Topics (sensor data)

```
sensors/temperature/{process_unit_id}
sensors/pressure/{process_unit_id}
sensors/flow/{process_unit_id}
sensors/vibration/{process_unit_id}
```

### Result Topics (per-channel anomaly scores)

```
ai/results/temperature/{process_unit_id}
ai/results/pressure/{process_unit_id}
ai/results/flow/{process_unit_id}
ai/results/vibration/{process_unit_id}
```

### Fused Result Topic

```
ai/results/fused/{process_unit_id}
```

The fused topic carries the combined score from the
`SensorFusionEngine` and is the recommended source for downstream
alerting rules.

### Example MQTT Configuration

```bash
# Subscribe to all sensor types across all process units
MQTT_TOPICS=sensors/+/+

# Results published to
MQTT_TOPIC_RESULTS=ai/results

# Fusion-specific settings
FUSION_STRATEGY=weighted_average
FUSION_GLOBAL_THRESHOLD=0.7
```

---

## Configuring Per-Sensor Thresholds

Different sensor types have different noise characteristics and
criticality levels.  Configure per-sensor thresholds via the
`ANOMALY_THRESHOLDS_JSON` environment variable:

```json
{
    "thresholds": {
        "temperature": 0.50,
        "pressure": 0.70,
        "vibration": 0.60,
        "flow": 0.65
    },
    "weights": {
        "temperature": 1.2,
        "pressure": 1.0,
        "vibration": 0.8,
        "flow": 0.9
    },
    "fusion_strategy": "weighted_average",
    "global_threshold": 0.7
}
```

### How to Tune

1. **Collect baseline data** — run the system for 24–48 hours on
   known-good production data.
2. **Observe score distributions** — per-channel and fused.
3. **Set channel thresholds** at the 99th percentile of normal scores
   for that channel.
4. **Set the global threshold** at the 99th percentile of fused scores.
5. **Iterate** — lower thresholds for safety-critical channels,
   raise them for noisy-but-non-critical sensors.

---

## Integration with AIO Dataflows for Cloud Alerting

The sensor fusion engine publishes results to the MQTT broker.  AIO
dataflows then route anomaly events to cloud services for alerting,
storage, and dashboarding.

### Route Fused Anomalies to Azure Event Hubs

```yaml
apiVersion: connectivity.iotoperations.azure.com/v1
kind: Dataflow
metadata:
  name: fused-anomalies-to-cloud
  namespace: azure-iot-operations
spec:
  operations:
    - operationType: source
      sourceSettings:
        endpointRef: aio-broker
        dataSources:
          - ai/results/fused/+
    - operationType: filter
      filterSettings:
        expression: "$payload.is_anomaly == true"
    - operationType: destination
      destinationSettings:
        endpointRef: eventhubs-endpoint
        dataDestination: process-anomalies
```

### Route Per-Channel Alerts for Diagnostics

```yaml
apiVersion: connectivity.iotoperations.azure.com/v1
kind: Dataflow
metadata:
  name: channel-alerts-to-adx
  namespace: azure-iot-operations
spec:
  operations:
    - operationType: source
      sourceSettings:
        endpointRef: aio-broker
        dataSources:
          - ai/results/+/+
    - operationType: filter
      filterSettings:
        expression: "$payload.anomaly_score > 0.85"
    - operationType: destination
      destinationSettings:
        endpointRef: adx-endpoint
        dataDestination: ChannelAnomalies
```

### End-to-End Alert Flow

```
Sensor → MQTT → Inference Server → MOMENT (per-channel)
                                      ↓
                               SensorFusionEngine
                                      ↓
                              MQTT (fused topic)
                                      ↓
                              AIO Dataflow (filter)
                                      ↓
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                   Event Hubs    ADX (KQL)    Logic App
                   (stream)      (analysis)   (notify)
                                                 ↓
                                        Teams / Email / SMS
                                        ServiceNow incident
```

---

## Example: Semiconductor Manufacturing (SECOM-style)

Using the [SECOM dataset](../../datasets/README.md) as a reference:

1. **590 sensor features** monitored per wafer lot
2. MOMENT processes each feature as a separate channel
3. Anomaly detection identifies lots likely to fail quality inspection
4. Alert lead time: 15–30 minutes before end-of-line test

This enables **early scrap detection** — diverting defective lots before
further (wasted) processing steps.

---

## Quick Start: Run the Demo

To see multi-sensor fusion in action without any infrastructure:

```bash
cd samples/03-end-to-end
pip install momentfm matplotlib numpy
python process_anomaly_demo.py
```

Or explore interactively in the Jupyter notebook:

```bash
cd notebooks
jupyter notebook 02-anomaly-detection-multisensor.ipynb
```
