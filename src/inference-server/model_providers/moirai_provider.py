"""MOIRAI model provider for aio-sensor-intelligence.

MOIRAI is a masked-encoder-based universal forecasting transformer from
Salesforce.  This provider wraps ``uni2ts.model.moirai.MoiraiForecast``
and ``MoiraiModule`` and exposes them through the unified
:class:`ModelProvider` interface.

Install with::

    pip install uni2ts
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
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
except ImportError:  # pragma: no cover
    MoiraiForecast = None  # type: ignore[assignment,misc]
    MoiraiModule = None  # type: ignore[assignment,misc]

try:
    from gluonts.dataset.common import ListDataset
except ImportError:  # pragma: no cover
    ListDataset = None  # type: ignore[assignment,misc]

from .base import TASK_FORECAST, ModelProvider, ModelResult

logger = logging.getLogger(__name__)

# Default forecast parameters — can be overridden via predict(**kwargs).
_DEFAULT_PREDICTION_LENGTH = 96
_DEFAULT_CONTEXT_LENGTH = 512
_DEFAULT_PATCH_SIZE = "auto"
_DEFAULT_NUM_SAMPLES = 100
_DEFAULT_BATCH_SIZE = 32


class MoiraiProvider(ModelProvider):
    """Provider backed by Salesforce MOIRAI (via the uni2ts library).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default ``Salesforce/moirai-1.1-R-large``).
    device : str
        ``"auto"`` to pick CUDA when available, or ``"cpu"``/``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/moirai-1.1-R-large",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._module: Any = None

    # ------------------------------------------------------------------
    # ModelProvider interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the MOIRAI module weights from HuggingFace."""
        if MoiraiModule is None or MoiraiForecast is None:
            raise ImportError(
                "uni2ts is not installed. "
                "Install with: pip install uni2ts"
            )
        if ListDataset is None:
            raise ImportError(
                "gluonts is not installed. "
                "Install with: pip install gluonts"
            )
        if torch is None:
            raise ImportError("PyTorch is required but not installed.")

        logger.info("Loading MOIRAI module from %s …", self.model_name)
        self._module = MoiraiModule.from_pretrained(self.model_name)
        logger.info(
            "MoiraiProvider loaded (%s) on %s", self.model_name, self.device
        )

    def predict(self, data: np.ndarray, task: str, **kwargs: Any) -> ModelResult:
        """Run forecasting inference with MOIRAI.

        Parameters
        ----------
        data : np.ndarray
            Shape ``[batch, channels, seq_len]``.
        task : str
            Must be ``"forecasting"``.
        **kwargs
            Optional overrides: ``prediction_length``, ``context_length``,
            ``patch_size``, ``num_samples``, ``batch_size``.
        """
        if task != TASK_FORECAST:
            raise ValueError(
                f"MoiraiProvider only supports '{TASK_FORECAST}', got '{task}'"
            )
        if self._module is None:
            raise RuntimeError("Model not loaded — call load() first.")

        prediction_length: int = kwargs.get(
            "prediction_length",
            kwargs.get("forecast_horizon", _DEFAULT_PREDICTION_LENGTH),
        )
        context_length: int = kwargs.get("context_length", _DEFAULT_CONTEXT_LENGTH)
        patch_size = kwargs.get("patch_size", _DEFAULT_PATCH_SIZE)
        num_samples: int = kwargs.get("num_samples", _DEFAULT_NUM_SAMPLES)
        batch_size: int = kwargs.get("batch_size", _DEFAULT_BATCH_SIZE)

        # data shape: [batch, channels, seq_len]
        n_batch, n_channels, seq_len = data.shape

        # Clamp context_length to actual sequence length.
        context_length = min(context_length, seq_len)

        # Build a GluonTS ListDataset from the input array.
        # Each entry represents one batch item; for multivariate data the
        # target has shape [channels, seq_len], for univariate [seq_len].
        import pandas as pd

        entries = []
        for i in range(n_batch):
            target = data[i]  # [channels, seq_len]
            if n_channels == 1:
                target = target.squeeze(0)  # [seq_len]
            entries.append(
                {"start": pd.Timestamp("2000-01-01"), "target": target}
            )
        ds = ListDataset(entries, freq="h")

        # Build the MoiraiForecast predictor.
        # NOTE: target_dim should match the number of variates per series.
        # feat_dynamic_real_dim and past_feat_dynamic_real_dim default to 0
        # when no exogenous features are provided.
        forecast_model = MoiraiForecast(
            module=self._module,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=n_channels,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = forecast_model.create_predictor(batch_size=batch_size)

        if self.device != "cpu" and torch is not None:
            predictor = predictor

        # Run prediction — returns an iterator of gluonts SampleForecast.
        forecasts = list(predictor.predict(ds))

        # Collect forecast medians into an array.
        # Each SampleForecast has .samples of shape [num_samples, prediction_length]
        # or [num_samples, target_dim, prediction_length] for multivariate.
        # We take the median across samples as the point forecast.
        results = []
        for fc in forecasts:
            # fc.mean gives the mean forecast; shape varies by dimensionality.
            results.append(fc.mean)

        result_np = np.stack(results, axis=0)  # [batch, ...]

        return ModelResult(
            values=result_np,
            task=task,
            metadata={
                "model": self.model_name,
                "device": self.device,
                "prediction_length": prediction_length,
                "context_length": context_length,
                "num_samples": num_samples,
            },
        )

    def supported_tasks(self) -> list[str]:
        return [TASK_FORECAST]

    def info(self) -> dict:
        return {
            "provider": "moirai",
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
