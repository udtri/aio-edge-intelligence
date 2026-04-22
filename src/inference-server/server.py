"""FastAPI inference server for edge AI sensor intelligence.

Provides REST endpoints for time-series inference tasks.

Exposes REST endpoints for anomaly detection, time-series forecasting, and
sensor-data classification backed by a pluggable model-provider system.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import AppConfig
from schemas import (
    AnomalyResult,
    ClassificationResult,
    ForecastResult,
    HealthResponse,
    ModelInfo,
    SensorData,
)
from tasks import AnomalyDetector, Forecaster

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Global state populated during startup
# ---------------------------------------------------------------------------
_config: AppConfig | None = None
_provider: Any = None  # model_providers.ModelProvider instance
_anomaly_detector: AnomalyDetector | None = None
_forecaster: Forecaster | None = None
_ready: bool = False


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load configuration and initialise the model provider on startup."""
    global _config, _provider, _anomaly_detector, _forecaster, _ready  # noqa: PLW0603

    _config = AppConfig()
    logger.info(
        "Configuration loaded — provider=%s  model=%s  device=%s",
        _config.model_provider,
        _config.model_name,
        _config.model_device,
    )

    try:
        from model_providers import get_provider

        _provider = get_provider(
            _config.model_provider,
            model_name=_config.model_name,
            device=_config.model_device,
        )
        _provider.load()
        _anomaly_detector = AnomalyDetector(
            provider=_provider,
            window_size=_config.window_size,
        )
        _forecaster = Forecaster(
            provider=_provider,
            window_size=_config.window_size,
        )
        _ready = True
        logger.info("Model provider '%s' is ready", _config.model_provider)
    except Exception:
        logger.exception("Failed to initialise model provider")
        _ready = False

    yield

    # Shutdown: allow provider to clean up if it exposes a close method.
    if _provider is not None and hasattr(_provider, "close"):
        _provider.close()
    logger.info("Inference server shut down")


app = FastAPI(
    title="Edge AI Sensor Intelligence — Inference Server",
    version="0.1.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_provider():
    """Raise 503 if the model provider is not available."""
    if not _ready or _provider is None:
        raise HTTPException(
            status_code=503,
            detail="Model provider is not loaded or not ready",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return current health status of the inference server."""
    return HealthResponse(
        status="healthy" if _ready else "degraded",
        model=_config.model_provider if _config else "unknown",
        ready=_ready,
    )


@app.get("/models", response_model=ModelInfo)
async def models() -> ModelInfo:
    """Return information about the currently loaded model provider."""
    if _config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    return ModelInfo(
        provider=_config.model_provider,
        model_name=_config.model_name,
        supported_tasks=_provider.supported_tasks() if _provider else ["anomaly_detection", "forecasting", "classification"],
        device=_config.model_device,
        status="ready" if _ready else "not_ready",
    )


@app.post("/infer/anomaly", response_model=AnomalyResult)
async def infer_anomaly(data: SensorData) -> AnomalyResult:
    """Run anomaly detection on the supplied sensor data."""
    _require_provider()
    try:
        import numpy as np
        values = np.array(data.values, dtype=np.float64)
        result: AnomalyResult = _anomaly_detector.detect(
            values, sensor_id=data.sensor_id
        )
        return result
    except Exception as exc:
        logger.exception("Anomaly detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class ForecastRequest(BaseModel):
    """Request body for the forecast endpoint."""

    data: SensorData
    forecast_horizon: int = Field(default=96, ge=1, description="Number of steps to forecast")


@app.post("/infer/forecast", response_model=ForecastResult)
async def infer_forecast(request: ForecastRequest) -> ForecastResult:
    """Produce a time-series forecast from the supplied sensor data."""
    _require_provider()
    try:
        import numpy as np
        values = np.array(request.data.values, dtype=np.float64)
        result: ForecastResult = _forecaster.forecast(
            values,
            horizon=request.forecast_horizon,
            sensor_id=request.data.sensor_id,
        )
        return result
    except Exception as exc:
        logger.exception("Forecasting failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/infer/classify", response_model=ClassificationResult)
async def infer_classify(data: SensorData) -> ClassificationResult:
    """Classify the supplied sensor data."""
    _require_provider()
    try:
        result: ClassificationResult = _provider.classify(data)
        return result
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
