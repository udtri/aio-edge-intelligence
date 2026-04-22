"""Forecasting task — wraps a ModelProvider for time-series prediction and RUL estimation.

Provides single-shot forecasting, probabilistic confidence intervals via
Monte-Carlo sampling (if the model supports it), and Remaining Useful Life
(RUL) estimation by projecting forecast trends against a threshold.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from model_providers.base import TASK_FORECAST, ModelProvider, ModelResult
from schemas import ForecastResult

logger = logging.getLogger(__name__)


class Forecaster:
    """High-level time-series forecaster built on a :class:`ModelProvider`.

    Parameters
    ----------
    provider : ModelProvider
        A loaded model provider that supports ``TASK_FORECAST``.
    default_horizon : int
        Default number of steps to forecast (default 96 for MOMENT).
    window_size : int
        Expected input window length (default 512).
    """

    def __init__(
        self,
        provider: ModelProvider,
        default_horizon: int = 96,
        window_size: int = 512,
    ) -> None:
        if TASK_FORECAST not in provider.supported_tasks():
            raise ValueError(
                f"Provider does not support '{TASK_FORECAST}'. "
                f"Supported: {provider.supported_tasks()}"
            )
        self.provider = provider
        self.default_horizon = default_horizon
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(
        self,
        data: np.ndarray,
        horizon: int | None = None,
        sensor_id: str | None = None,
    ) -> ForecastResult:
        """Produce a point forecast.

        Parameters
        ----------
        data : np.ndarray
            Input history.  Shape ``(seq_len,)`` or ``(batch, channels, seq_len)``.
        horizon : int or None
            Forecast horizon (steps ahead).  Defaults to ``self.default_horizon``.
        sensor_id : str or None
            Optional identifier for result tagging.

        Returns
        -------
        ForecastResult
        """
        horizon = horizon or self.default_horizon
        data = self._prepare_input(data)
        result: ModelResult = self.provider.predict(
            data, task=TASK_FORECAST, forecast_horizon=horizon
        )
        values = result.values.flatten()[:horizon]

        return ForecastResult(
            sensor_id=sensor_id,
            forecast_values=values.tolist(),
            forecast_horizon=horizon,
            timestamp=datetime.utcnow(),
        )

    def forecast_with_confidence(
        self,
        data: np.ndarray,
        horizon: int | None = None,
        n_samples: int = 100,
        sensor_id: str | None = None,
    ) -> dict:
        """Produce a forecast with confidence intervals via Monte-Carlo sampling.

        If the model is deterministic, light Gaussian noise is injected
        into the input to simulate uncertainty.

        Parameters
        ----------
        data : np.ndarray
            Input history (1-D or provider-format).
        horizon : int or None
            Steps ahead.
        n_samples : int
            Number of forward passes for the confidence envelope.
        sensor_id : str or None
            Optional sensor identifier.

        Returns
        -------
        dict
            Keys: ``forecast`` (:class:`ForecastResult` with the median),
            ``lower_95``, ``upper_95``, ``lower_50``, ``upper_50`` (lists
            of floats), and ``std`` (per-step standard deviation).
        """
        horizon = horizon or self.default_horizon
        data_3d = self._prepare_input(data)

        samples: list[np.ndarray] = []
        # Estimate input noise scale from the data for perturbation
        noise_scale = float(np.std(data_3d)) * 0.01

        for _ in range(n_samples):
            perturbed = data_3d + np.random.randn(*data_3d.shape) * noise_scale
            result = self.provider.predict(
                perturbed, task=TASK_FORECAST, forecast_horizon=horizon
            )
            samples.append(result.values.flatten()[:horizon])

        stacked = np.stack(samples, axis=0)  # (n_samples, horizon)
        median = np.median(stacked, axis=0)
        std = np.std(stacked, axis=0)

        return {
            "forecast": ForecastResult(
                sensor_id=sensor_id,
                forecast_values=median.tolist(),
                forecast_horizon=horizon,
                timestamp=datetime.utcnow(),
            ),
            "lower_95": np.percentile(stacked, 2.5, axis=0).tolist(),
            "upper_95": np.percentile(stacked, 97.5, axis=0).tolist(),
            "lower_50": np.percentile(stacked, 25.0, axis=0).tolist(),
            "upper_50": np.percentile(stacked, 75.0, axis=0).tolist(),
            "std": std.tolist(),
        }

    def estimate_rul(
        self,
        data: np.ndarray,
        failure_threshold: float,
        horizon: int | None = None,
        sampling_rate_hz: float = 10.0,
    ) -> dict:
        """Estimate Remaining Useful Life by projecting the forecast trend.

        Forecasts the signal forward and finds the first step where the
        predicted value exceeds *failure_threshold*.  If the threshold is
        not breached within the forecast horizon, the RUL is reported as
        ``> horizon`` steps.

        Parameters
        ----------
        data : np.ndarray
            Recent sensor history.
        failure_threshold : float
            The value above which the equipment is considered failed.
        horizon : int or None
            Maximum look-ahead (defaults to ``self.default_horizon``).
        sampling_rate_hz : float
            Sampling rate of the sensor data (Hz) — used to convert
            steps to seconds.

        Returns
        -------
        dict
            ``rul_steps`` (int or None), ``rul_seconds`` (float or None),
            ``threshold_breached`` (bool), ``forecast`` (:class:`ForecastResult`).
        """
        horizon = horizon or self.default_horizon
        fc = self.forecast(data, horizon=horizon)
        values = np.array(fc.forecast_values)

        breach_indices = np.where(values >= failure_threshold)[0]
        if len(breach_indices) > 0:
            rul_steps = int(breach_indices[0])
            rul_seconds = rul_steps / sampling_rate_hz
            breached = True
        else:
            rul_steps = None
            rul_seconds = None
            breached = False

        return {
            "rul_steps": rul_steps,
            "rul_seconds": rul_seconds,
            "threshold_breached": breached,
            "forecast": fc,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, data: np.ndarray) -> np.ndarray:
        """Reshape input to ``(batch, channels, seq_len)``."""
        if data.ndim == 1:
            return data.reshape(1, 1, -1)
        if data.ndim == 2:
            return data[np.newaxis, ...]
        return data
