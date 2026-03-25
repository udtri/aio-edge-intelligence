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

## Configuring Alert Thresholds

Thresholds control when anomaly detections become actionable alerts.
Configure them based on your process tolerance and desired
sensitivity.

### Global Threshold

Set via environment variable:
```bash
ANOMALY_THRESHOLD=0.6
```

Any anomaly score above this value is flagged as `is_anomaly: true` in
the result payload.

### Per-Sensor Threshold (Advanced)

For processes where different sensor types have different criticality:

```json
{
    "thresholds": {
        "temperature": 0.5,
        "pressure": 0.7,
        "vibration": 0.6,
        "flow": 0.65
    }
}
```

Pass as `ANOMALY_THRESHOLDS_JSON` environment variable.

### Tuning Guidance

| If you get … | Adjust … |
|---|---|
| Too many false alarms | Raise threshold (e.g., 0.6 → 0.75) |
| Missed real anomalies | Lower threshold (e.g., 0.6 → 0.45) |
| Alerts too late | Reduce STRIDE for more frequent checks |
| Alerts too noisy | Increase WINDOW_SIZE for more context |

> **Tip:** Start with the default threshold (0.6), run for 24–48 hours
> on known-good production data, and observe the score distribution.
> Set the threshold at the 99th percentile of normal scores.

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

## Integration with AIO Dataflows for Cloud Alerting

### Route All Anomalies to Azure Event Hubs

```yaml
apiVersion: connectivity.iotoperations.azure.com/v1
kind: Dataflow
metadata:
  name: anomalies-to-cloud
  namespace: azure-iot-operations
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
        dataDestination: process-anomalies
```

### Route Critical Alerts to Azure Data Explorer for Analysis

```yaml
apiVersion: connectivity.iotoperations.azure.com/v1
kind: Dataflow
metadata:
  name: critical-to-adx
  namespace: azure-iot-operations
spec:
  operations:
    - operationType: source
      sourceSettings:
        endpointRef: aio-broker
        dataSources:
          - ai/results
    - operationType: filter
      filterSettings:
        expression: "$payload.anomaly_score > 0.85"
    - operationType: destination
      destinationSettings:
        endpointRef: adx-endpoint
        dataDestination: CriticalAnomalies
```

### Trigger Azure Logic App for Operator Notification

Combine the Event Hubs destination with an Azure Logic App that sends:
- Email / SMS alerts to operators
- Microsoft Teams channel notifications
- ServiceNow incident creation

---

## Example: Semiconductor Manufacturing (SECOM-style)

Using the [SECOM dataset](../../datasets/README.md) as a reference:

1. **590 sensor features** monitored per wafer lot
2. MOMENT processes each feature as a separate channel
3. Anomaly detection identifies lots likely to fail quality inspection
4. Alert lead time: 15–30 minutes before end-of-line test

This enables **early scrap detection** — diverting defective lots before
further (wasted) processing steps.
