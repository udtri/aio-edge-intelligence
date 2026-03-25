# aio-sensor-intelligence

**Physical AI at the Edge with Azure IoT Operations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Deploy open-source time-series foundation models — **MOMENT**, **MOIRAI**, and **Chronos** — on any Kubernetes cluster for predictive maintenance, anomaly detection, and time-series forecasting on factory sensor data. Optionally integrate with **Azure IoT Operations (AIO)** for production-grade OPC UA ingestion and dataflows.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Any Kubernetes Cluster (K3s, AKS, AKS EE, etc.)                   │
│                                                                     │
│  ┌──────────────┐    ┌────────────────┐    ┌────────────────────┐  │
│  │ AIO OPC UA   │───▶│ MQTT Broker    │───▶│ Inference Server   │  │
│  │ Connector    │    │ AIO-Connected: │    │ (Pod)              │  │
│  │ (AIO mode)   │    │  aio-broker    │    │  - Model Provider  │  │
│  └──────────────┘    │ Standalone:    │    │    interface       │  │
│                      │  mosquitto     │    │  - Anomaly detect  │  │
│  ┌──────────────┐    │                │    │  - Forecasting     │  │
│  │ Sensor       │───▶│ Topics:        │◀──▶│  - Classification  │  │
│  │ Simulator    │    │  sensors/*     │    │                    │  │
│  │ (Pod)        │    │  ai/results    │    │  Results → MQTT    │  │
│  └──────────────┘    └────────────────┘    └────────────────────┘  │
│                            │                                        │
│                      ┌─────▼──────────┐    ┌────────────────────┐  │
│                      │ AIO Dataflows  │───▶│ Grafana Dashboard  │  │
│                      │ (AIO mode only)│    │ (optional)         │  │
│                      └────────────────┘    └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow:** Factory sensors (or the built-in simulator) publish readings over MQTT. The inference server subscribes, runs the configured model, and publishes results (anomaly scores, forecasts, classifications) back to MQTT. In AIO-Connected mode, dataflows can route results to the cloud or a Grafana dashboard.

---

## Two Deployment Modes

| | Standalone Mode | AIO-Connected Mode |
|---|---|---|
| **Infrastructure** | Docker Compose or plain K8s | K8s cluster with Azure IoT Operations |
| **MQTT Broker** | Mosquitto | AIO MQTT Broker |
| **Sensor Ingestion** | Built-in simulator | AIO OPC UA Connector + real factory sensors |
| **Dataflows** | — | AIO Dataflows for routing & transformation |
| **Best For** | Local dev, demos, quick evaluation | Production edge deployments |

---

## Supported Models

| Provider | Model | Best For | Config |
|----------|-------|----------|--------|
| MOMENT | AutonLab/MOMENT-1-large | Anomaly detection, classification | `MODEL_PROVIDER=moment` |
| MOIRAI | Salesforce/moirai-1.1-R-large | Universal forecasting | `MODEL_PROVIDER=moirai` |
| Chronos | amazon/chronos-t5-large | Production forecasting | `MODEL_PROVIDER=chronos` |
| Custom | Your own model | Any task | `MODEL_PROVIDER=custom` |

All models are swappable at runtime through the **ModelProvider** interface — set `MODEL_PROVIDER` in your environment or Helm values and the inference server loads the corresponding provider. Implementing a custom provider requires a single Python class.

---

## Quick Start

### Standalone Mode (Docker Compose)

```bash
# Clone and run with docker-compose
git clone https://github.com/YOUR_ORG/aio-sensor-intelligence.git
cd aio-sensor-intelligence
docker compose -f deploy/standalone/docker-compose.yaml up
```

This starts Mosquitto, the sensor simulator, and the inference server with the default model. Sensor data flows on `sensors/*` topics; results appear on `ai/results`.

### AIO-Connected Mode (Helm)

```bash
# Prerequisites: K8s cluster with AIO installed
helm install aio-si ./deploy/aio-connected/helm/aio-sensor-intelligence
```

The Helm chart deploys the inference server and configures AIO dataflows to subscribe to sensor topics from the AIO MQTT broker. Ensure your AIO OPC UA connector is already publishing data.

---

## Use Cases

- **Predictive Maintenance** — Detect bearing wear, motor degradation, or pump cavitation before failure using anomaly detection models.
- **Process Anomaly Detection** — Identify out-of-spec conditions in real time across temperature, pressure, and vibration sensors.
- **Time-Series Forecasting** — Forecast sensor trends to optimize scheduling, inventory, and energy consumption.

---

## Project Structure

```
aio-sensor-intelligence/
├── deploy/
│   ├── standalone/              # Docker Compose & plain K8s manifests
│   │   └── docker-compose.yaml
│   └── aio-connected/           # AIO-integrated deployment
│       ├── helm/
│       │   └── aio-sensor-intelligence/
│       └── dataflows/
├── src/
│   ├── inference/               # Inference server
│   │   ├── server.py
│   │   └── providers/           # Pluggable model providers
│   │       ├── base.py          # ModelProvider interface
│   │       ├── moment.py
│   │       ├── moirai.py
│   │       ├── chronos.py
│   │       └── custom.py
│   └── simulator/               # Sensor data simulator
├── config/                      # Default configuration files
├── dashboards/                  # Grafana dashboard definitions
├── tests/
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Configuration

Key environment variables for the inference server:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PROVIDER` | Model backend (`moment`, `moirai`, `chronos`, `custom`) | `moment` |
| `MQTT_BROKER_HOST` | MQTT broker hostname | `localhost` |
| `MQTT_BROKER_PORT` | MQTT broker port | `1883` |
| `SUBSCRIBE_TOPICS` | Comma-separated MQTT topics to subscribe to | `sensors/#` |
| `PUBLISH_TOPIC` | Topic for inference results | `ai/results` |

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Roadmap

- **VLA / VLM Vision Integration** — Extend the ModelProvider interface to support Vision-Language-Action and Vision-Language models for visual inspection and robotic control at the edge.
- **Additional model providers** — TimesFM, PatchTST, and more.
- **Edge-optimized inference** — ONNX Runtime and quantized model support for constrained hardware.

---

## License

This project is licensed under the [MIT License](LICENSE).
