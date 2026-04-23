# Foundation Models vs. Traditional ML for Time Series

## Why This Document Exists

After deploying and testing the MOMENT time-series foundation model on an Azure IoT
Operations cluster, a natural question emerges: **why not just use scikit-learn?**

Random forests, XGBoost, logistic regression, ARIMA — these are proven, well-understood,
and fast. They work. So what do encoder-based foundation models like MOMENT
actually change?

This document captures the honest answer, grounded in what we observed during live
deployment and testing on a 3-node Arc-enabled K3s cluster.

---

## The Short Answer

Traditional ML is better when you have **one well-defined problem, good labels, and
stable data**. Foundation models are better when you have **many problems, few labels,
and shifting data** — which is the reality of most industrial edge deployments.

The shift isn't about accuracy. It's about **time-to-value and coverage at scale**.

---

## Where Traditional ML Still Wins

| Scenario | Why Traditional ML Is Better |
|----------|------------------------------|
| Single sensor, thousands of labeled examples | A tuned XGBoost will be more accurate, faster, and smaller |
| Well-characterized failure modes | Decision trees give explainable rules: "if vibration > 5g AND temp > 80°C → bearing fault" |
| Inference speed matters (microseconds) | Traditional models run in µs; MOMENT needs ~600ms |
| Memory-constrained devices | Random forest = KBs; MOMENT = 2.5 GB |
| Regulatory explainability required | "Feature X exceeded threshold Y" is auditable; "reconstruction error was high" is harder to audit |

**If you have a single sensor on a single machine, 10,000 labeled examples, and a
stable data distribution — don't use MOMENT. Use scikit-learn.**

---

## Where Traditional ML Breaks Down

### 1. The Labeling Bottleneck

Traditional ML requires labeled training data per machine, per failure mode, per
sensor configuration.

```
Factory with 200 CNC machines:

Traditional approach:
  Machine 1:  Collect 6 months of data → label 47 anomalies → engineer features
              → train model → validate → deploy                         [works]
  Machine 2:  Different sensor config → new feature pipeline → retrain  [works]
  Machine 3:  New machine type → start from scratch                     [6 weeks]
  ...
  Machine 200: No one has labeled this yet                              [nothing]

  Result: 3 machines covered after 6 months of ML engineering

Foundation model approach:
  All 200 machines: Send 512 raw values → get anomaly scores
  Zero training, zero labels, zero per-machine tuning
  Day 1 coverage for all 200 machines

  Result: 200 machines covered on deployment day
```

The accuracy on Machine 1 may be lower than a tuned XGBoost. But having any coverage
on Machines 4–200 is infinitely better than having none.

**This is the fundamental value proposition: zero-shot generalization trades per-task
accuracy for universal coverage.**

### 2. The Feature Engineering Tax

For traditional ML on time series, the real work isn't choosing the algorithm — it's
**engineering the features**:

```python
# What a traditional time-series ML pipeline looks like
# (weeks of domain expert effort per sensor type)

features = {
    # Statistical features
    "mean": np.mean(window),
    "std": np.std(window),
    "skewness": scipy.stats.skew(window),
    "kurtosis": scipy.stats.kurtosis(window),
    "percentile_95": np.percentile(window, 95),

    # Frequency domain (FFT)
    "dominant_freq": get_dominant_frequency(window),
    "spectral_entropy": compute_spectral_entropy(window),
    "band_energy_0_100hz": bandpass_energy(window, 0, 100),
    "band_energy_100_500hz": bandpass_energy(window, 100, 500),

    # Trend features
    "linear_slope": linregress(window).slope,
    "rolling_mean_diff": rolling_mean(window, 50) - rolling_mean(window, 200),

    # Autocorrelation
    "acf_lag_1": autocorrelation(window, lag=1),
    "acf_lag_10": autocorrelation(window, lag=10),

    # Domain-specific (vibration)
    "rms_velocity": compute_rms_velocity(window),
    "crest_factor": max(abs(window)) / rms(window),
    "bearing_defect_freq_energy": bpfo_energy(window, shaft_rpm, n_balls),

    # ... 30-50 more features, tuned per sensor type
}

model.predict(pd.DataFrame([features]))
```

Different sensor types need different features. Vibration needs FFT and bearing
geometry. Temperature needs trend and rate-of-change. Pressure needs derivative and
threshold features. A domain expert spends weeks per sensor type figuring out which
features matter.

Foundation models replace all of this:

```python
# What the foundation model pipeline looks like
scores = model(x_enc=raw_512_values, input_mask=mask)
```

The transformer learns its own internal features from raw signal via self-attention
and patching. This is the same shift that happened in NLP — people used to hand-craft
TF-IDF and bag-of-words features; now transformers learn representations directly from
text.

**The feature engineering that took weeks is now encoded in the pretrained weights.**

### 3. Open-World vs. Closed-World Detection

Traditional classifiers operate in a **closed world** — they detect only the failure
modes they were trained on:

```
Training data contains: bearing_wear, misalignment, imbalance
Model can detect:       bearing_wear, misalignment, imbalance

What about:
  - A failure mode never seen before?              → classified as "normal"
  - A new machine type just installed?             → model doesn't fit
  - A subtle degradation over 6 months?            → below learned thresholds
  - A combination of conditions never co-occurred? → not in training distribution
```

Foundation models operate in an **open world** — they detect anything that deviates
from general temporal patterns:

```
MOMENT doesn't know about bearing_wear or misalignment.
It knows: "this signal doesn't look like normal time-series behavior."

Novel failure mode?     → reconstruction error goes up → detected
New machine type?       → still works (it learned general patterns, not your specific machine)
Subtle degradation?     → reconstruction error slowly rises → detected as trend
Novel combination?      → the combination produces unusual temporal patterns → detected
```

**Closed-world = "Is this one of the 5 failures I know?"**
**Open-world = "Does this look normal?"**

### 4. The MLOps Treadmill

Traditional models degrade when data distributions shift:

```
January:   Train model on winter HVAC data → deploy → works great
April:     Spring patterns → model accuracy drops → retrain
July:      Summer load → model degrades again → retrain
September: New firmware changes sensor sampling rate → features break → rebuild pipeline
November:  Sensor replaced with different model → calibration shift → retrain

This is the MLOps treadmill:
  Train → Deploy → Drift → Detect drift → Collect new data → Relabel →
  Retrain → Validate → Redeploy → Works → Drifts again → ...

  Cost: continuous ML engineering effort, forever
```

Foundation models are more robust to distribution shift because they learned general
temporal patterns from millions of diverse time series — not your specific data
distribution. They don't memorize your baselines; they understand what time series
in general look like.

**The MLOps cost shifts from "retrain monthly" to "deploy once, fine-tune if needed."**

### 5. One Model vs. Fifty

A real industrial deployment:

```
Traditional:
  Temperature anomaly detector    → trained model A, feature pipeline A
  Vibration anomaly detector      → trained model B, feature pipeline B
  Pressure forecaster             → trained model C, feature pipeline C
  Multi-sensor classifier         → trained model D, feature pipeline D
  Each: different features, different algorithm, different hyperparams

  Operational cost: 4+ models × N machine types × seasonal retraining
  = dozens of ML pipelines to maintain

Foundation model:
  One model binary (2.5 GB)
  Four task heads: anomaly, forecast, classify, embed
  Same weights serve all sensor types
  No per-task training for baseline capability

  Operational cost: one deployment, one upgrade path
```

---

## The Complexity Gap: What Transformers See That Trees Don't

### Multi-Scale Temporal Patterns

A random forest sees the features you engineered. A transformer sees the raw signal
and learns features at multiple time scales simultaneously through self-attention:

```
Window of 512 vibration samples:

What XGBoost sees (your engineered features):
  mean=2.3, std=0.8, dominant_freq=47Hz, kurtosis=3.2
  → a fixed-length snapshot, loses temporal structure

What MOMENT sees (via patching + attention):
  Patch 1 (samples 0-63):    local trend, micro-oscillation
  Patch 2 (samples 64-127):  relationship to patch 1, emerging pattern
  ...
  Patch 8 (samples 448-511): long-range dependency back to patch 1
  + cross-patch attention:    how the pattern in samples 0-63
                              relates to the pattern in samples 400-511
```

The transformer captures **relationships between different parts of the signal** that
you'd need hand-crafted features to approximate.

### Cross-Channel Relationships (Multivariate)

When you send `[1, 3, 512]` (temperature, pressure, vibration):

```
Traditional: train separate models per sensor, then manually write correlation rules
  if temp_anomaly AND pressure_anomaly → alert
  (brittle, misses subtle correlations)

Foundation model: processes all 3 channels jointly
  Attention across channels captures:
  "Temperature rising while pressure drops" = heat exchanger fouling
  "Vibration increasing with temperature" = bearing degradation
  "Pressure oscillating at new frequency" = valve instability
  (learned from pretraining data, not hand-coded rules)
```

### Non-Stationarity

Real-world sensors are non-stationary — the statistics change over time. Traditional
models assume stationarity (or require explicit differencing, detrending).

Foundation models handle this natively because they've seen non-stationary patterns
during pretraining — trends, level shifts, seasonal changes, regime transitions.

---

## How This Changes MLOps

### Traditional ML Lifecycle

```
Problem definition     → 2 weeks
Data collection        → 4 weeks (often need 6+ months of history)
Data labeling          → 2-8 weeks (domain expert time)
Feature engineering    → 2-4 weeks per sensor type
Model selection        → 1-2 weeks (grid search, cross-validation)
Training + validation  → 1-2 weeks
Deployment             → 1-2 weeks (infra, monitoring, A/B test)
Monitoring + retrain   → ongoing, every 1-3 months

Time to first value: 3-6 months
Ongoing cost: continuous ML engineering
```

### Foundation Model Lifecycle

```
Deploy inference server      → 1 day (Helm install)
Send raw sensor data         → immediate (MQTT publish)
Get anomaly scores           → 512 samples later (~minutes)
Evaluate zero-shot quality   → 1-2 days

If zero-shot is good enough  → done
If need better accuracy      → fine-tune on your data (1-2 weeks)

Time to first value: 1-2 days
Ongoing cost: model upgrades (swap to newer version)
```

**The paradigm shifts from "build a model for your problem" to "evaluate a pretrained
model on your problem."** Data collection and labeling become optional refinements
rather than prerequisites.

---

## When To Use What: A Decision Framework

```
START
  │
  ├─ Do you have labeled training data for this specific problem?
  │   │
  │   ├─ YES, thousands of labeled examples
  │   │   │
  │   │   ├─ Is the data distribution stable (not shifting seasonally)?
  │   │   │   │
  │   │   │   ├─ YES → Use traditional ML (XGBoost, Random Forest)
  │   │   │   │        Best accuracy, fastest inference, most explainable
  │   │   │   │
  │   │   │   └─ NO  → Foundation model as baseline + fine-tune
  │   │   │            More robust to drift, less retraining
  │   │   │
  │   │   └─ Do you need to detect novel (unseen) failure modes?
  │   │       │
  │   │       ├─ YES → Foundation model (open-world detection)
  │   │       └─ NO  → Traditional ML is fine
  │   │
  │   └─ NO, few or zero labels
  │       │
  │       └─ Foundation model (zero-shot)
  │          Only viable option without labels
  │
  ├─ How many different sensor types / machines do you have?
  │   │
  │   ├─ 1-5 well-understood machines → Traditional ML (affordable to label + tune each)
  │   └─ 50+ diverse machines        → Foundation model (can't build 50 pipelines)
  │
  ├─ What latency do you need?
  │   │
  │   ├─ Microseconds (real-time control loop) → Traditional ML or threshold rules
  │   └─ Hundreds of milliseconds (monitoring) → Foundation model is fine
  │
  └─ What explainability do you need?
      │
      ├─ Regulatory audit trail → Traditional ML (clear feature-based decisions)
      └─ Operational alerting  → Foundation model (anomaly score + location is sufficient)
```

---

## The Pragmatic Hybrid Approach

The best real-world deployments use both:

```
Layer 1 (Foundation model — broad coverage):
  Deploy MOMENT across ALL sensors, ALL machines
  Zero-shot anomaly detection everywhere
  Catches unknown unknowns, provides baseline coverage
  Cost: one deployment

Layer 2 (Traditional ML — precision where it matters):
  For the 5 most critical machines, train specific models
  XGBoost with domain-specific features
  Higher accuracy, better explainability for these machines
  Cost: ML engineering for 5 machines

Layer 3 (Fine-tuned foundation model — best of both):
  For machines where zero-shot isn't accurate enough
  but you don't have enough labels for traditional ML,
  fine-tune MOMENT on your limited labeled data
  Cost: moderate (1-2 weeks per model)
```

This gives you:
- **Universal coverage** from day one (foundation model)
- **Precision** where it matters most (traditional ML)
- **Cost-effective scaling** for the middle tier (fine-tuned foundation model)

---

## Comparison: MOMENT Field-Test Results vs. Typical Traditional ML

Based on our live deployment on the AIO cluster:

| Metric | MOMENT (Zero-Shot) | Traditional ML (Tuned) |
|--------|-------------------|----------------------|
| **Setup time** | 1 day (deploy + test) | 3-6 months (collect, label, train) |
| **Labels required** | Zero | Thousands |
| **Feature engineering** | None | Weeks of domain expertise |
| **Anomaly detection accuracy** | Spike region scored 7.8× higher than normal | Typically >95% on known failure modes |
| **Novel failure detection** | Yes (open-world) | No (closed-world) |
| **Inference latency** | ~600ms (CPU) | ~1ms |
| **Memory** | 2.5 GB | KBs–MBs |
| **Determinism** | Bit-for-bit identical | Depends on algorithm |
| **Multi-task** | Anomaly + forecast + classify + embed | One model per task |
| **Handles distribution shift** | More robust (general patterns) | Degrades, needs retrain |
| **Explainability** | Per-sample scores (where, not why) | Feature importance (what and why) |

---

## The Bigger Picture: Why This Matters Now

Time-series encoder foundation models represent the same inflection point that NLP
saw with BERT in 2018. Before BERT, every NLP task needed task-specific training.
After BERT, pretrained models provided strong baselines for any text task — and
fine-tuning surpassed custom models.

The same pattern is playing out for time series:

```
2018-2022: Every sensor anomaly problem → custom feature engineering + custom model
2023-2025: Foundation models (MOMENT, TimesFM) → pretrained baseline for any time series
2025+:     Fine-tuned foundation models → surpass custom models with less effort
```

Models like MOMENT are
**generalizing** what it means to understand temporal patterns — across domains,
across sensor types, across industries. The model that learned from electrical grid
data can detect anomalies in HVAC systems because the underlying temporal patterns
(trends, seasonality, spikes, level shifts) are universal.

**The era of building a bespoke ML pipeline for every sensor on every machine is
ending. The era of deploying a general-purpose time-series intelligence layer —
and fine-tuning only where needed — is beginning.**

---

## Further Reading

- [MOMENT Model Behavior](moment-model-behavior.md) — field-tested I/O contracts, latency, determinism
- [Deployment Guide](deployment-guide-arc-k3s.md) — step-by-step record of deploying to Arc-enabled K3s
- [Anomaly Detection Use Case](use-cases/anomaly-detection.md) — multi-sensor fusion for manufacturing
- [Predictive Maintenance Use Case](use-cases/predictive-maintenance.md) — rotating machinery degradation
- [MOMENT paper](https://arxiv.org/abs/2402.03885) — "MOMENT: A Family of Open Time-series Foundation Models"
