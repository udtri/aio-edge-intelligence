"""Google TimesFM provider for zero-shot time-series forecasting.

The provider keeps TimesFM behind the same ``ModelProvider`` contract as the
rest of the service. TimesFM 2.5 is the default because its weights use the
Apache-2.0 license. TimesFM 3.0 can be selected explicitly for multivariate
research, but its published checkpoint is currently non-commercial.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import timesfm
except ImportError:  # pragma: no cover
    timesfm = None  # type: ignore[assignment]

try:
    from timesfm3 import TimesFM3Forecaster
except ImportError:  # pragma: no cover
    TimesFM3Forecaster = None  # type: ignore[assignment,misc]

from .base import TASK_FORECAST, ModelProvider, ModelResult

logger = logging.getLogger(__name__)

DEFAULT_TIMESFM_MODEL = "google/timesfm-2.5-200m-pytorch"
TIMESFM_3_MODEL = "google/timesfm-3.0-pytorch"
_TIMESFM_3_MARKER = "timesfm-3"


class TimesFMProvider(ModelProvider):
    """Forecast with Google Research's TimesFM 2.5 or TimesFM 3.0.

    Input arrays use the repository-wide ``[batch, channels, sequence]``
    convention. TimesFM 2.5 forecasts each channel independently. TimesFM 3.0
    jointly forecasts channels and can also receive covariates through
    :meth:`predict` keyword arguments.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TIMESFM_MODEL,
        device: str = "auto",
        max_context: int = 1024,
        max_horizon: int = 256,
        per_core_batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.per_core_batch_size = per_core_batch_size
        self._model: Any = None

    @property
    def is_timesfm_3(self) -> bool:
        """Return whether the configured checkpoint uses the 3.x API."""
        return _TIMESFM_3_MARKER in self.model_name.lower()

    def load(self) -> None:
        """Load the configured checkpoint and initialize its inference path."""
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install 'timesfm[torch]'")

        if self.is_timesfm_3:
            self._load_timesfm_3()
        else:
            self._load_timesfm_2p5()

        logger.info(
            "TimesFMProvider loaded (%s) on %s",
            self.model_name,
            self.device,
        )

    def predict(self, data: np.ndarray, task: str, **kwargs: Any) -> ModelResult:
        """Produce a point forecast and retain quantiles in result metadata."""
        if task != TASK_FORECAST:
            raise ValueError(
                f"TimesFM supports only '{TASK_FORECAST}', received '{task}'."
            )
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() before predict().")

        values = np.asarray(data, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                "TimesFM input must have shape [batch, channels, sequence]; "
                f"received {values.shape}."
            )

        horizon = int(kwargs.pop("forecast_horizon", self.max_horizon))
        if horizon < 1:
            raise ValueError("forecast_horizon must be at least 1.")
        if not self.is_timesfm_3 and horizon > self.max_horizon:
            raise ValueError(
                f"forecast_horizon {horizon} exceeds compiled maximum "
                f"{self.max_horizon}."
            )

        if self.is_timesfm_3:
            point, quantiles = self._predict_timesfm_3(values, horizon, **kwargs)
        else:
            if kwargs:
                names = ", ".join(sorted(kwargs))
                raise ValueError(f"TimesFM 2.5 does not accept provider options: {names}")
            point, quantiles = self._predict_timesfm_2p5(values, horizon)

        return ModelResult(
            values=point,
            task=task,
            metadata={
                "model": self.model_name,
                "device": self.device,
                "quantiles": quantiles,
                "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "native_multivariate": self.is_timesfm_3,
            },
        )

    def supported_tasks(self) -> list[str]:
        return [TASK_FORECAST]

    def info(self) -> dict:
        return {
            "provider": "timesfm",
            "model_name": self.model_name,
            "device": self.device,
            "supported_tasks": self.supported_tasks(),
            "native_multivariate": self.is_timesfm_3,
            "weights_license": (
                "timesfm-non-commercial-license-v1.0"
                if self.is_timesfm_3
                else "Apache-2.0"
            ),
        }

    def _load_timesfm_2p5(self) -> None:
        if timesfm is None or not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise ImportError(
                "TimesFM 2.5 API is unavailable. "
                "Install with: pip install 'timesfm[torch]>=3.0.1,<4'"
            )

        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.model_name
        )
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                per_core_batch_size=self.per_core_batch_size,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def _load_timesfm_3(self) -> None:
        if TimesFM3Forecaster is None:
            raise ImportError(
                "TimesFM 3 API is unavailable. "
                "Install with: pip install 'timesfm[torch]>=3.0.1,<4'"
            )

        logger.warning(
            "TimesFM 3.0 checkpoint weights are restricted to non-commercial, "
            "non-production use under their current license."
        )
        self._model = TimesFM3Forecaster.from_pretrained(
            self.model_name,
            device=self.device,
            per_core_batch_size=self.per_core_batch_size,
        )

    def _predict_timesfm_2p5(
        self,
        values: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size, channels, _ = values.shape
        inputs = [values[b, c] for b in range(batch_size) for c in range(channels)]
        point, quantiles = self._model.forecast(horizon=horizon, inputs=inputs)

        point_array = np.asarray(point, dtype=np.float32).reshape(
            batch_size, channels, horizon
        )
        quantile_array = np.asarray(quantiles, dtype=np.float32)
        quantile_array = quantile_array[..., 1:].reshape(
            batch_size, channels, horizon, -1
        )
        return point_array, quantile_array

    def _predict_timesfm_3(
        self,
        values: np.ndarray,
        horizon: int,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        allowed = {"past_only_covariates", "past_future_covariates"}
        unknown = set(kwargs) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown TimesFM 3 provider options: {names}")

        outputs = list(
            self._model.predict_batch(
                contexts=[sample for sample in values],
                horizon=horizon,
                past_only_covariates=kwargs.get("past_only_covariates"),
                past_future_covariates=kwargs.get("past_future_covariates"),
                return_quantiles=True,
                use_symmetric_averaging=False,
                make_positive=False,
            )
        )
        point_items = []
        quantile_items = []
        for output in outputs:
            point_item = np.asarray(output.forecast, dtype=np.float32)
            quantile_item = np.asarray(output.quantiles, dtype=np.float32)
            if point_item.ndim == 1:
                point_item = point_item[np.newaxis, :]
            if quantile_item.ndim == 2:
                quantile_item = quantile_item[np.newaxis, :, :]
            point_items.append(point_item)
            quantile_items.append(quantile_item)

        point = np.stack(point_items, axis=0)
        quantiles = np.stack(quantile_items, axis=0)
        return point, quantiles

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        return device
