"""Chronos model provider for aio-sensor-intelligence.

Chronos is a family of pretrained time-series forecasting models from
Amazon.  This provider wraps ``chronos.ChronosPipeline`` and exposes it
through the unified :class:`ModelProvider` interface.

Install with::

    pip install chronos-forecasting
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
    from chronos import ChronosPipeline
except ImportError:  # pragma: no cover
    ChronosPipeline = None  # type: ignore[assignment,misc]

from .base import TASK_FORECAST, ModelProvider, ModelResult

logger = logging.getLogger(__name__)

# Default forecast parameters — can be overridden via predict(**kwargs).
_DEFAULT_PREDICTION_LENGTH = 96
_DEFAULT_NUM_SAMPLES = 100


class ChronosProvider(ModelProvider):
    """Provider backed by Amazon Chronos (via the chronos-forecasting library).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default ``amazon/chronos-t5-large``).
    device : str
        ``"auto"`` to pick CUDA when available, or ``"cpu"``/``"cuda"``.
    torch_dtype : str
        Torch dtype string (default ``"float32"``).  Use ``"bfloat16"`` on
        hardware that supports it for faster inference.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-large",
        device: str = "auto",
        torch_dtype: str = "float32",
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.torch_dtype = torch_dtype
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # ModelProvider interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the Chronos pipeline from HuggingFace."""
        if ChronosPipeline is None:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Install with: pip install chronos-forecasting"
            )
        if torch is None:
            raise ImportError("PyTorch is required but not installed.")

        logger.info("Loading Chronos pipeline from %s …", self.model_name)
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )
        logger.info(
            "ChronosProvider loaded (%s) on %s", self.model_name, self.device
        )

    def predict(self, data: np.ndarray, task: str, **kwargs: Any) -> ModelResult:
        """Run forecasting inference with Chronos.

        Parameters
        ----------
        data : np.ndarray
            Shape ``[batch, channels, seq_len]``.  Chronos is a univariate
            model, so each channel is forecast independently.
        task : str
            Must be ``"forecasting"``.
        **kwargs
            Optional overrides: ``prediction_length`` (or ``forecast_horizon``),
            ``num_samples``.
        """
        if task != TASK_FORECAST:
            raise ValueError(
                f"ChronosProvider only supports '{TASK_FORECAST}', got '{task}'"
            )
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded — call load() first.")
        if torch is None:
            raise ImportError("PyTorch is required but not installed.")

        prediction_length: int = kwargs.get(
            "prediction_length",
            kwargs.get("forecast_horizon", _DEFAULT_PREDICTION_LENGTH),
        )
        num_samples: int = kwargs.get("num_samples", _DEFAULT_NUM_SAMPLES)

        # data shape: [batch, channels, seq_len]
        n_batch, n_channels, seq_len = data.shape

        # Chronos operates on univariate series.  Flatten batch × channels
        # into a list of 1-D tensors, run prediction, then reshape back.
        series_list: list[torch.Tensor] = []
        for b in range(n_batch):
            for c in range(n_channels):
                series_list.append(
                    torch.tensor(data[b, c, :], dtype=torch.float32)
                )

        # pipeline.predict accepts a list of 1-D tensors and returns a tensor
        # of shape [total_series, num_samples, prediction_length].
        forecast_samples = self._pipeline.predict(
            series_list,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )  # [n_batch * n_channels, num_samples, prediction_length]

        # Take the median across samples as the point forecast.
        # Shape: [n_batch * n_channels, prediction_length]
        point_forecast = torch.median(forecast_samples, dim=1).values.cpu().numpy()

        # Reshape back to [batch, channels, prediction_length].
        result_np = point_forecast.reshape(n_batch, n_channels, prediction_length)

        return ModelResult(
            values=result_np,
            task=task,
            metadata={
                "model": self.model_name,
                "device": self.device,
                "prediction_length": prediction_length,
                "num_samples": num_samples,
            },
        )

    def supported_tasks(self) -> list[str]:
        return [TASK_FORECAST]

    def info(self) -> dict:
        return {
            "provider": "chronos",
            "model_name": self.model_name,
            "device": self.device,
            "supported_tasks": self.supported_tasks(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
