# Contributing to AIO Edge Intelligence

> [!NOTE]
> This is a personal, experimental project — not an official Microsoft repository.

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Adding a New Model Provider](#adding-a-new-model-provider)
- [Adding a New Sensor Profile](#adding-a-new-sensor-profile)
- [Adding a New Use Case](#adding-a-new-use-case)
- [Testing Guidelines](#testing-guidelines)
- [Building & Testing Docker Images Locally](#building--testing-docker-images-locally)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)

---

## Development Environment Setup

### Prerequisites

- Python 3.10+
- Docker (for container builds)
- Helm 3 (for chart development)
- kubectl (for Kubernetes testing)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/udtri/aio-edge-intelligence.git
cd aio-sensor-intelligence

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify the installation
ruff check src/
pytest
```

---

## Project Structure

```
aio-sensor-intelligence/
├── src/
│   ├── inference-server/       # FastAPI inference server
│   │   ├── model_providers/    # Pluggable AI model backends
│   │   │   ├── base.py         # ModelProvider ABC + shared types
│   │   │   ├── moment_provider.py
│   │   │   └── custom_provider.py
│   │   ├── tasks/              # AI task implementations
│   │   ├── server.py           # FastAPI application entrypoint
│   │   ├── config.py           # Configuration management
│   │   └── Dockerfile
│   ├── sensor-simulator/       # Synthetic sensor data generator
│   │   ├── profiles/           # Sensor profile plugins
│   │   │   ├── vibration_motor.py
│   │   │   ├── temperature_furnace.py
│   │   │   ├── pressure_hydraulic.py
│   │   │   └── audio_bearing.py
│   │   ├── simulator.py        # Main simulator entrypoint
│   │   └── Dockerfile
│   └── dashboard/              # Monitoring dashboard
├── deploy/
│   ├── standalone/             # Standalone K8s manifests + docker-compose
│   └── aio-connected/          # Azure IoT Operations integration
│       ├── helm/               # Helm chart
│       ├── kustomize/          # Kustomize overlays (dev/prod)
│       └── dataflows/          # AIO dataflow definitions
├── datasets/                   # Sample data and benchmarks
├── notebooks/                  # Jupyter notebooks for exploration
├── docs/                       # Documentation
└── pyproject.toml              # Project metadata and dependencies
```

---

## Adding a New Model Provider

Model providers implement the `ModelProvider` abstract base class defined in
`src/inference-server/model_providers/base.py`.

### Step-by-Step

1. **Create the provider file** in `src/inference-server/model_providers/`:

   ```python
   # src/inference-server/model_providers/my_provider.py
   """My custom model provider."""

   import numpy as np
   from model_providers.base import ModelProvider, ModelResult, TASK_FORECAST, TASK_ANOMALY


   class MyModelProvider(ModelProvider):
       """Provider for the My Model foundation model."""

       def __init__(self, model_name: str = "my-org/my-model", device: str = "cpu"):
           self.model_name = model_name
           self.device = device
           self._model = None

       def load(self) -> None:
           """Load model weights."""
           # Import your model library and load weights
           # self._model = MyModel.from_pretrained(self.model_name)
           pass

       def predict(self, data: np.ndarray, task: str, **kwargs) -> ModelResult:
           """Run inference on the input data."""
           # Implement inference logic for each supported task
           if task == TASK_ANOMALY:
               scores = self._run_anomaly_detection(data)
               return ModelResult(values=scores, task=task)
           elif task == TASK_FORECAST:
               horizon = kwargs.get("horizon", 96)
               forecast = self._run_forecast(data, horizon)
               return ModelResult(values=forecast, task=task)
           raise ValueError(f"Unsupported task: {task}")

       def supported_tasks(self) -> list[str]:
           return [TASK_ANOMALY, TASK_FORECAST]

       def info(self) -> dict:
           return {
               "name": "my-model",
               "version": "1.0",
               "tasks": self.supported_tasks(),
           }
   ```

2. **Register the provider** in `src/inference-server/model_providers/__init__.py`.

3. **Add optional dependencies** (if any) to `pyproject.toml` under `[project.optional-dependencies]`.

4. **Test the provider** — write unit tests that verify `load()`, `predict()`, `supported_tasks()`, and `info()` work correctly.

### Provider Interface

| Method             | Description                                       |
| ------------------ | ------------------------------------------------- |
| `load()`           | Load model weights and initialize runtime state   |
| `predict(data, task, **kwargs)` | Run inference; return a `ModelResult` |
| `supported_tasks()` | Return list of task name strings                 |
| `info()`           | Return metadata dict (name, version, etc.)        |

---

## Adding a New Sensor Profile

Sensor profiles generate realistic synthetic time-series data for testing.

### Step-by-Step

1. **Create the profile file** in `src/sensor-simulator/profiles/`:

   ```python
   # src/sensor-simulator/profiles/my_sensor.py
   """My custom sensor profile."""

   import numpy as np


   class MySensorProfile:
       """Simulates readings from a custom sensor type.

       Args:
           sampling_rate: Samples per second (Hz).
       """

       def __init__(self, sampling_rate: float = 100.0) -> None:
           self.sampling_rate = sampling_rate
           self._rng = np.random.default_rng()

       def generate(self, n_samples: int) -> np.ndarray:
           """Generate n_samples of sensor data."""
           t = np.arange(n_samples) / self.sampling_rate
           signal = np.sin(2 * np.pi * 1.0 * t)  # Replace with realistic model
           signal += self._rng.normal(0, 0.01, size=n_samples)
           return signal

       def inject_anomaly(self, anomaly_type: str) -> None:
           """Queue an anomaly for the next generate() call."""
           pass
   ```

2. **Register the profile** in `src/sensor-simulator/profiles/__init__.py`:

   ```python
   from profiles.my_sensor import MySensorProfile

   # Add to _REGISTRY:
   _REGISTRY["my_sensor"] = MySensorProfile
   ```

3. **Test the profile**:

   ```python
   from profiles import get_profile
   profile = get_profile("my_sensor")
   data = profile.generate(512)
   assert data.shape == (512,)
   ```

---

## Adding a New Use Case

Use cases combine a sensor profile with model inference for a specific industrial scenario.

1. **Define the sensor profile** (see above) that captures the physics of your domain.
2. **Choose or create a model provider** suited to the use case.
3. **Configure the deployment** — update `deploy/standalone/k8s/configmap.yaml` with appropriate topics and model settings.
4. **Add a notebook** in `notebooks/` demonstrating the end-to-end workflow.
5. **Document the use case** in `docs/` with architecture diagrams and expected results.

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_providers.py

# Run with coverage (install pytest-cov first)
pytest --cov=src --cov-report=term-missing
```

### Smoke Testing Sensor Profiles

```bash
python -c "
from src.sensor_simulator.profiles import get_profile
import numpy as np

for name in ['vibration', 'temperature', 'pressure', 'audio']:
    data = get_profile(name).generate(512)
    assert isinstance(data, np.ndarray)
    assert data.shape == (512,)
    print(f'{name}: OK')
"
```

### What to Test

- **Model providers**: Test `load()`, `predict()` with sample data, `supported_tasks()`, `info()`.
- **Sensor profiles**: Test `generate()` returns the correct shape and finite values.
- **API endpoints**: Use `httpx.AsyncClient` with the FastAPI `TestClient`.
- **Configuration**: Validate that config loading handles missing/invalid values gracefully.

---

## Building & Testing Docker Images Locally

### Inference Server

```bash
# CPU build (default)
cd src/inference-server
docker build -t inference-server:local .

# CUDA build
docker build --build-arg RUNTIME=cuda -t inference-server:cuda .

# Run locally
docker run -p 8080:8080 inference-server:local
curl http://localhost:8080/health
```

### Sensor Simulator

```bash
cd src/sensor-simulator
docker build -t sensor-simulator:local .

# Run (needs MQTT broker)
docker run -e MQTT_BROKER_HOST=host.docker.internal sensor-simulator:local
```

### Full Stack with Docker Compose

```bash
cd deploy/standalone
docker-compose up --build
```

---

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

### Configuration

See `pyproject.toml` for Ruff settings:

- **Line length**: 100 characters
- **Target Python**: 3.10+

### Quick Commands

```bash
# Lint
ruff check src/

# Auto-fix lint issues
ruff check --fix src/

# Format
ruff format src/

# Check formatting without changes
ruff format --check src/
```

### Conventions

- Use **type hints** for all function signatures.
- Use **docstrings** (Google or NumPy style) for all public classes and functions.
- Prefer `from __future__ import annotations` for modern type syntax.
- Use `logging` instead of `print()` for runtime messages.

---

## Pull Request Process

1. **Fork and branch** — create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** — follow the code style and testing guidelines above.

3. **Run CI checks locally** before pushing:
   ```bash
   ruff check src/
   ruff format --check src/
   pytest
   ```

4. **Push and open a PR** against `main`:
   - Provide a clear title and description.
   - Reference any related issues (e.g., `Closes #42`).
   - Ensure all CI checks pass.

5. **Code review** — address reviewer feedback. All PRs require at least one approval.

6. **Merge** — maintainers will squash-merge once approved and CI is green.
