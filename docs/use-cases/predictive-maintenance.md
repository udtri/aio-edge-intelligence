# Use Case: Predictive Maintenance for Rotating Machinery

## Scenario

Industrial facilities rely on **rotating machinery** — motors, pumps,
compressors, turbines, and fans — that are critical to production.
Unexpected failures cause costly downtime, safety hazards, and cascading
damage.

Traditional **preventive maintenance** uses fixed schedules (e.g.,
replace bearings every 6 months) regardless of actual condition.  This
leads to:

- Over-maintenance (replacing healthy components)
- Under-maintenance (missing early degradation)

**Predictive maintenance** monitors the *actual condition* of equipment
in real time and triggers maintenance only when degradation is detected.

---

## Sensor Types

| Sensor | What It Measures | Degradation Indicators |
|---|---|---|
| **Vibration** (accelerometer) | Mechanical oscillation | Imbalance, misalignment, bearing wear, looseness |
| **Temperature** (thermocouple / RTD) | Bearing / winding heat | Friction increase, lubrication breakdown, overload |
| **Acoustic emission** | High-frequency sound | Micro-cracks, cavitation, electrical arcing |
| **Current** (CT sensor) | Motor current draw | Rotor bar faults, load anomalies |

Vibration is the most information-rich signal for rotating machinery and
is the primary input for this use case.

---

## How MOMENT Detects Degradation

MOMENT is a **time-series foundation model** pre-trained on hundreds of
thousands of time-series datasets.  It understands general temporal
patterns without needing to be trained on *your specific machine*.

### Anomaly Detection (Reconstruction)

1. The inference server feeds a **512-sample sliding window** of
   vibration data into MOMENT's reconstruction head.
2. MOMENT reconstructs what the signal *should* look like based on
   learned temporal patterns.
3. The **reconstruction error** (difference between actual and
   reconstructed) is computed.
4. Healthy operation → low error.  Degradation → elevated error,
   especially at frequencies associated with bearing defects or
   imbalance.

### Forecasting

1. MOMENT's forecasting head predicts the next **96 samples** based on
   the current window.
2. If the forecast diverges significantly from subsequent actual
   readings, it indicates a **regime change** (e.g., bearing entering
   failure mode).
3. Forecast trends can predict temperature rise or vibration growth
   before it reaches critical levels.

### Degradation Timeline

```
Normal operation       Early degradation      Advanced degradation    Failure
      │                      │                        │                  │
      ▼                      ▼                        ▼                  ▼
  Low anomaly score    Score rising (0.3-0.5)   Score high (0.7-0.9)   Spike (>0.95)
  No alerts            Warning alerts           Critical alerts        Machine stop
                       ◄── Maintenance window ──►
```

The goal is to detect the transition from "normal" to "early
degradation" — giving operators a **maintenance window** of days to
weeks before failure.

---

## Setup

### 1. Connect Vibration Sensors

Configure sensors to publish to MQTT:

```
Topic:   sensors/vibration/{machine_id}
Payload: {
    "timestamp": "2025-01-15T10:30:00.000Z",
    "sensor_id": "pump-bearing-01",
    "vibration": 2.345,
    "temperature": 62.1
}
Rate:    10–100 Hz (10 Hz recommended for MOMENT)
```

### 2. Configure the Inference Server

```bash
# Environment variables
MQTT_TOPICS=sensors/vibration/#
WINDOW_SIZE=512
STRIDE=64
ANOMALY_THRESHOLD=0.7
TASKS=anomaly_detection,forecasting
```

### 3. Deploy

**Standalone:**
```bash
docker compose up -d
```

**AIO-connected:**
```bash
helm install aio-sensor-intelligence deploy/helm/aio-sensor-intelligence \
    --set mqtt.topics="sensors/vibration/#" \
    --set inference.tasks="anomaly_detection\,forecasting"
```

### 4. Monitor Results

Subscribe to the results topic:
```bash
mosquitto_sub -t "ai/results" -v
```

Or use the end-to-end test script:
```bash
python samples/03-end-to-end/run_e2e.py
```

---

## Interpreting Anomaly Scores

| Score Range | Interpretation | Recommended Action |
|---|---|---|
| **0.0 – 0.3** | Normal operation | No action needed |
| **0.3 – 0.5** | Minor deviation | Monitor; check at next scheduled stop |
| **0.5 – 0.7** | Notable anomaly | Investigate within days |
| **0.7 – 0.9** | Significant degradation | Plan maintenance within 24–48 hours |
| **0.9 – 1.0** | Critical / imminent failure | Immediate maintenance or controlled shutdown |

> **Important:** These thresholds are starting points.  Calibrate them
> for your specific equipment by observing scores during known-good and
> known-degraded operation.

---

## Expected Results

With vibration data from a motor with developing bearing wear:

- **Days 1–5:** Anomaly scores hover around 0.1–0.2 (normal).
- **Day 6:** Scores start climbing to 0.3–0.4 as bearing defect
  frequency emerges.
- **Day 8:** Scores reach 0.6–0.7; the forecasting head predicts
  continued growth.
- **Day 9:** Alert threshold crossed; maintenance scheduled.
- **Day 12 (would-be failure):** Machine is already repaired.

In testing with the [CWRU Bearing Dataset](../datasets/README.md),
MOMENT correctly identifies bearing fault signatures across inner race,
outer race, and ball defect types — with zero fine-tuning.

---

## Using the Task Pipeline (Code Examples)

### Anomaly Detection

```python
from collections import deque
import numpy as np

from model_providers.moment_provider import MomentProvider
from tasks import AnomalyDetector

# Initialise provider and detector
provider = MomentProvider(device="cpu")
provider.load()

detector = AnomalyDetector(provider, threshold=0.7)

# --- Batch detection on a NumPy array ---
window = np.random.randn(512)  # replace with real sensor data
result = detector.detect(window, sensor_id="pump-bearing-01")
print(result.is_anomaly)      # True / False
print(result.severity)        # "normal", "warning", or "critical"
print(result.anomaly_scores)  # per-timestep scores

# --- Streaming detection with a sliding window ---
buffer = deque(maxlen=512)
for sample in sensor_stream():          # your real-time data source
    buffer.append(sample)
    if len(buffer) == 512:
        result = detector.detect_stream(buffer, sensor_id="pump-bearing-01")
        if result.is_anomaly:
            publish_alert(result)        # push to MQTT / dashboard

# --- Adaptive thresholding ---
# After processing several windows the detector tracks rolling stats:
stats = detector.get_rolling_stats()
print(f"Adaptive threshold: {stats['adaptive_threshold']:.3f}")
```

### Forecasting & Remaining Useful Life

```python
from tasks import Forecaster

forecaster = Forecaster(provider, default_horizon=96)

# Point forecast
fc = forecaster.forecast(window, sensor_id="pump-bearing-01")
print(fc.forecast_values)      # list of 96 predicted values

# Forecast with 95 % confidence interval
ci = forecaster.forecast_with_confidence(window, horizon=96, n_samples=50)
print(ci["lower_95"])          # 2.5th percentile
print(ci["upper_95"])          # 97.5th percentile

# Remaining Useful Life estimation
rul = forecaster.estimate_rul(
    window,
    failure_threshold=5.0,     # vibration threshold in g
    sampling_rate_hz=10.0,
)
if rul["threshold_breached"]:
    print(f"Estimated RUL: {rul['rul_seconds']:.0f} seconds")
else:
    print("No failure predicted within forecast horizon")
```

---

## MQTT Topic Configuration

| Topic | Direction | Description |
|---|---|---|
| `sensors/vibration/{machine_id}` | **Inbound** | Raw vibration data from sensors |
| `sensors/temperature/{machine_id}` | **Inbound** | Bearing/winding temperature |
| `ai/results` | **Outbound** | Anomaly scores, forecasts, severity |
| `ai/alerts/warning` | **Outbound** | Scores in the warning band (0.5–0.7) |
| `ai/alerts/critical` | **Outbound** | Scores in the critical band (≥ 0.7) |
| `ai/rul/{machine_id}` | **Outbound** | RUL estimates for each machine |

### Inbound payload schema

```json
{
    "timestamp": "2025-01-15T10:30:00.000Z",
    "sensor_id": "pump-bearing-01",
    "values": [2.345, 2.351, 2.338, "..."],
    "channels": 1,
    "metadata": { "unit": "g", "sampling_rate_hz": 10 }
}
```

### Outbound alert payload schema

```json
{
    "sensor_id": "pump-bearing-01",
    "anomaly_scores": [0.12, 0.15, "...", 0.78],
    "threshold": 0.7,
    "is_anomaly": true,
    "severity": "critical",
    "timestamp": "2025-01-15T10:30:05.120Z"
}
```

---

## Anomaly Score Ranges & Severity Mapping

| Score Range | Severity | What It Means | Recommended Action |
|---|---|---|---|
| **0.0 – 0.3** | `normal` | Reconstruction error is within normal variation. Signal matches learned healthy patterns. | No action needed. |
| **0.3 – 0.5** | `normal` | Slight deviation. Often caused by load changes or minor environmental shifts. | Continue monitoring. |
| **0.5 – 0.7** | `warning` | Sustained elevated error. Emerging fault signatures (e.g., new spectral peaks at bearing defect frequencies). | Investigate within 48 hours; schedule inspection. |
| **0.7 – 0.9** | `critical` | Strong fault signature. Model reconstruction diverges significantly from actual signal. | Plan maintenance within 24 hours. |
| **0.9 – 1.0** | `critical` | Imminent failure. Broadband energy increase, impact events, or signal saturation. | Immediate controlled shutdown and repair. |

> **Tip:** Use ``AnomalyDetector.get_rolling_stats()`` to see the
> adaptive threshold the system has learned from your data.  Start with
> the defaults and adjust ``severity_thresholds`` after observing your
> equipment under known-good and known-bad conditions.

---

## Setting Up Alerting Thresholds

### Static thresholds (simple)

```python
# Use a fixed threshold — good for well-characterised equipment
detector = AnomalyDetector(provider, threshold=0.65)
```

### Adaptive thresholds (recommended)

```python
# Let the detector learn the baseline from the first N windows
detector = AnomalyDetector(provider)  # no fixed threshold

# During a known-healthy warm-up period, run detect() on each window.
# The detector accumulates rolling statistics automatically.
for window in healthy_warmup_windows:
    detector.detect(window)

# After warm-up, the auto-threshold adapts: mean + 2·std of scores.
stats = detector.get_rolling_stats()
print(f"Learned baseline mean: {stats['mean']:.4f}")
print(f"Adaptive threshold:    {stats['adaptive_threshold']:.4f}")
```

### Custom severity bands

```python
from tasks.anomaly_detection import AnomalyDetector, Severity

detector = AnomalyDetector(
    provider,
    severity_thresholds={
        Severity.NORMAL:   (0.0, 0.4),
        Severity.WARNING:  (0.4, 0.6),
        Severity.CRITICAL: (0.6, 1.0),
    },
)
```

### Routing alerts by severity

```bash
# Environment variables for the inference server
ALERT_TOPIC_WARNING=ai/alerts/warning
ALERT_TOPIC_CRITICAL=ai/alerts/critical
ANOMALY_THRESHOLD_WARNING=0.5
ANOMALY_THRESHOLD_CRITICAL=0.7
```

---

## Integration with AIO Dataflows

Route critical alerts to the cloud for fleet-wide visibility:

```yaml
apiVersion: connectivity.iotoperations.azure.com/v1
kind: Dataflow
metadata:
  name: maintenance-alerts
spec:
  operations:
    - operationType: source
      sourceSettings:
        endpointRef: aio-broker
        dataSources:
          - ai/results
    - operationType: filter
      filterSettings:
        expression: "$payload.anomaly_score > 0.7"
    - operationType: destination
      destinationSettings:
        endpointRef: eventhubs-endpoint
        dataDestination: maintenance-alerts
```
