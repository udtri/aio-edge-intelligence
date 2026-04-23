# MOMENT Model Behavior — Field-Tested Reference

This document captures the empirically validated behavior of the **MOMENT-1-large**
time-series foundation model (`AutonLab/MOMENT-1-large`) as deployed on an Azure IoT
Operations (AIO) cluster via the `aio-edge-intelligence` inference server.

All observations below come from live testing on a 3-node Arc-enabled K3s cluster
running AIO with CPU-only inference.

---

## Model Identity

| Property | Value |
|----------|-------|
| **Model ID** | `AutonLab/MOMENT-1-large` |
| **Library** | `momentfm==0.1.4` |
| **Backbone** | Google Flan-T5-Large transformer encoder |
| **Parameters** | ~350M |
| **Download size** | ~1.4 GB (HuggingFace) |
| **RAM footprint** | ~2–3 GB at inference |
| **Load time** | ~12 s (cached), ~2 min (first download) |

---

## How You Interact With It

MOMENT is **not** an LLM. It is a **deterministic regression model** for time-series
data.

| Aspect | MOMENT (Time Series) | LLM (e.g. GPT) |
|--------|---------------------|-----------------|
| **Input** | Fixed-size numeric tensor `[B, C, seq_len]` | Variable-length token sequence |
| **Output** | Fixed-size numeric tensor (same shape as input) | Variable-length token sequence |
| **Output predictability** | 100 % deterministic, known size | Non-deterministic, variable length |
| **Interaction** | Single HTTP POST → single JSON response | May require streaming / chunking |
| **Latency** | ~500–700 ms per inference (CPU) | ~500 ms–30 s depending on output |
| **Protocol** | HTTP POST **or** MQTT pub/sub | HTTP POST (often streaming) |
| **Temperature / sampling** | N/A — pure regression | Configurable sampling parameters |

### Key Implication

When you send 512 floats, you get back **exactly 512 floats**. There is no
variable-length generation, no token budget, no stop tokens. The response size
is fully predictable before you send the request.

---

## Input Shape

```
[batch_size, n_channels, seq_len]
```

- **Typical**: `[1, 1, 512]` — one univariate time series of 512 samples
- **Multivariate**: `[1, N, 512]` — N sensor channels, each 512 samples
- **Native context window**: 512 samples (but the model handles other lengths)

### Flexibility (Empirically Verified)

| Input Length | Behavior | Output Length |
|-------------|----------|---------------|
| 256 samples | ✅ Works | 256 scores |
| 512 samples | ✅ Native | 512 scores |
| 1024 samples | ✅ Works | 1024 scores |

The inference server pads/truncates internally. You don't need to worry about
exact window sizing.

---

## MOMENT Calling Convention

MOMENT uses **keyword-only** arguments — not positional:

```python
# ✅ Correct
output = pipeline(x_enc=tensor, input_mask=mask)

# ❌ Wrong — raises TypeError
output = pipeline(tensor)
```

The `forward()` signature is:

```python
def forward(self, *, x_enc: Tensor, input_mask: Tensor = None, mask: Tensor = None, **kwargs)
```

### Task Routing

The pretrained model always loads with `task_name='reconstruction'` regardless of
what you pass to `from_pretrained()`. To use different tasks, either:

1. Set `pipeline.task_name = 'forecasting'` before calling `pipeline()`
2. Call the task method directly: `pipeline.forecast(x_enc=..., input_mask=...)`

---

## Output by Task

### Anomaly Detection (Reconstruction)

The model reconstructs the input signal. Anomaly scores are computed as the
per-sample absolute error between original and reconstructed signals.

```
Input:  [1, 1, 512]  →  Output: .reconstruction  →  [1, 1, 512]
```

**Response from `/infer/anomaly`:**

```json
{
  "sensor_id": "furnace-01",
  "anomaly_scores": [0.001, 0.003, ..., 0.45, ...],   // 512 floats
  "threshold": 0.142,
  "is_anomaly": true,
  "severity": "normal",
  "timestamp": "2026-04-22T04:15:00Z"
}
```

**Empirical results:**

| Signal Type | Avg Score | Max Score | Threshold | Detected? |
|-------------|-----------|-----------|-----------|-----------|
| Clean sine wave | 0.061 | 0.170 | 0.142 | — |
| Sine + spike (3σ at pos 350–380) | 0.039 overall, **0.218 in spike region** | 0.499 | 0.164 | ✅ Spike region 7.8× higher |

### Forecasting

With the pretrained `PretrainHead`, the model produces a 512-length reconstruction
and the first `forecast_horizon` values are returned as the forecast.

```
Input:  [1, 1, 512]  →  Output: .forecast  →  [1, 1, 512]  →  slice [:horizon]
```

**Response from `/infer/forecast`:**

```json
{
  "sensor_id": "temp-01",
  "forecast_values": [-0.069, -0.037, ..., -0.769],   // exactly 96 floats
  "forecast_horizon": 96,
  "timestamp": "2026-04-22T04:16:00Z"
}
```

> **Note**: For true horizon-specific forecasting (output shape `[1, 1, horizon]`),
> the model requires fine-tuning with a `ForecastingHead`. The pretrained model
> uses a `PretrainHead` that always outputs 512 values.

### Classification

```
Input:  [1, N, 512]  →  Output: .logits  →  [1, num_classes]
```

### Imputation

```
Input:  [1, 1, 512] + mask  →  Output: .reconstruction  →  [1, 1, 512]
```

---

## Determinism

**Fully deterministic** in `model.eval()` mode:

- Same input → **bit-for-bit identical** output
- Max difference across repeated calls: `0.000000000000000`
- No sampling, no temperature, no randomness
- This is fundamentally different from LLMs

---

## Performance (CPU, 3-node K3s)

| Metric | Value |
|--------|-------|
| Inference latency (512 samples) | **~600 ms** |
| Inference latency (256 samples) | ~500 ms |
| Inference latency (1024 samples) | ~720 ms |
| Model load time | ~12 s (with HF cache) |
| First-inference delay (MQTT) | ~512 s at 1 Hz publish (waiting for 512 samples) |
| Memory usage | ~2.5 GB |
| CPU usage during inference | ~1.5 cores (burst) |

---

## REST API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/health` | GET | — | `{"status", "model", "ready"}` |
| `/models` | GET | — | `{"provider", "model_name", "supported_tasks", "device", "status"}` |
| `/infer/anomaly` | POST | `{"values", "channels", "sensor_id"}` | `{"anomaly_scores", "threshold", "is_anomaly", "severity"}` |
| `/infer/forecast` | POST | `{"data": {"values", "channels", "sensor_id"}, "forecast_horizon"}` | `{"forecast_values", "forecast_horizon"}` |
| `/infer/classify` | POST | `{"values", "channels", "sensor_id"}` | Classification result |

---

## MQTT Interface

The inference server also operates in MQTT pub/sub mode:

- **Subscribe**: `sensors/{type}/{device_id}` — receives sensor data
- **Publish**: `ai/results/{sensor_id}` — publishes inference results
- **Buffer**: Accumulates 512 samples per sensor before running inference
- **Stride**: After initial fill, runs inference every 64 new samples

