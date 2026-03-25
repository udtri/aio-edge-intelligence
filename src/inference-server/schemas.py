"""Pydantic models for request/response schemas used by the inference server."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SensorData(BaseModel):
    """Incoming sensor data payload."""

    timestamp: datetime | None = None
    values: list[float] = Field(..., description="Raw sensor values")
    channels: int = Field(default=1, ge=1, description="Number of sensor channels")
    sensor_id: str | None = None
    metadata: dict | None = None


class AnomalyResult(BaseModel):
    """Result of anomaly detection inference."""

    sensor_id: str | None = None
    anomaly_scores: list[float]
    threshold: float
    is_anomaly: bool
    severity: str = Field(
        default="normal",
        description="Severity level: normal, warning, or critical",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ForecastResult(BaseModel):
    """Result of time-series forecasting inference."""

    sensor_id: str | None = None
    forecast_values: list[float]
    forecast_horizon: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ClassificationResult(BaseModel):
    """Result of sensor-data classification inference."""

    sensor_id: str | None = None
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    """Metadata about the currently loaded model provider."""

    provider: str
    model_name: str
    supported_tasks: list[str]
    device: str
    status: str


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    model: str
    ready: bool
