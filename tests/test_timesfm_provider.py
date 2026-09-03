"""Unit tests for the TimesFM provider without downloading model weights."""

from __future__ import annotations

from types import SimpleNamespace

import model_providers.timesfm_provider as provider_module
import numpy as np
import pytest
from model_providers.base import TASK_FORECAST
from model_providers.timesfm_provider import TimesFMProvider


class _FakeTimesFM2p5Model:
    def __init__(self) -> None:
        self.compile_config = None

    def compile(self, config) -> None:
        self.compile_config = config

    def forecast(self, horizon: int, inputs: list[np.ndarray]):
        point = np.stack(
            [np.repeat(series[-1], horizon) for series in inputs], axis=0
        )
        quantiles = np.repeat(point[..., None], 10, axis=-1)
        return point, quantiles


class _FakeTimesFM2p5Class:
    last_model: _FakeTimesFM2p5Model | None = None

    @classmethod
    def from_pretrained(cls, model_name: str) -> _FakeTimesFM2p5Model:
        assert model_name == "google/timesfm-2.5-200m-pytorch"
        cls.last_model = _FakeTimesFM2p5Model()
        return cls.last_model


class _FakeTimesFM3Forecaster:
    def __init__(self) -> None:
        self.calls = []

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        assert model_name == "google/timesfm-3.0-pytorch"
        model = cls()
        model.load_options = kwargs
        return model

    def predict_batch(self, contexts, horizon: int, **kwargs):
        self.calls.append((contexts, horizon, kwargs))
        for context in contexts:
            channels = np.atleast_2d(context).shape[0]
            yield SimpleNamespace(
                forecast=np.ones((channels, horizon), dtype=np.float32),
                quantiles=np.ones((channels, horizon, 9), dtype=np.float32),
            )


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch):
    monkeypatch.setattr(provider_module, "torch", SimpleNamespace())


def test_timesfm_2p5_flattens_channels_and_restores_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        TimesFM_2p5_200M_torch=_FakeTimesFM2p5Class,
        ForecastConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(provider_module, "timesfm", fake_module)

    provider = TimesFMProvider(device="cpu")
    provider.load()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    result = provider.predict(data, TASK_FORECAST, forecast_horizon=5)

    assert result.values.shape == (2, 3, 5)
    assert result.metadata["quantiles"].shape == (2, 3, 5, 9)
    assert result.metadata["native_multivariate"] is False
    assert provider.info()["weights_license"] == "Apache-2.0"


def test_timesfm_3_preserves_multivariate_batches(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_module, "TimesFM3Forecaster", _FakeTimesFM3Forecaster
    )
    provider = TimesFMProvider(
        model_name="google/timesfm-3.0-pytorch",
        device="cpu",
    )
    provider.load()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    result = provider.predict(data, TASK_FORECAST, forecast_horizon=6)

    assert result.values.shape == (2, 3, 6)
    assert result.metadata["quantiles"].shape == (2, 3, 6, 9)
    assert result.metadata["native_multivariate"] is True
    assert provider.info()["weights_license"] == (
        "timesfm-non-commercial-license-v1.0"
    )


def test_timesfm_rejects_non_forecasting_tasks() -> None:
    provider = TimesFMProvider(device="cpu")
    provider._model = object()

    with pytest.raises(ValueError, match="supports only"):
        provider.predict(
            np.zeros((1, 1, 8), dtype=np.float32),
            "anomaly_detection",
        )


def test_timesfm_2p5_enforces_compiled_horizon() -> None:
    provider = TimesFMProvider(device="cpu", max_horizon=8)
    provider._model = object()

    with pytest.raises(ValueError, match="compiled maximum"):
        provider.predict(
            np.zeros((1, 1, 8), dtype=np.float32),
            TASK_FORECAST,
            forecast_horizon=9,
        )
