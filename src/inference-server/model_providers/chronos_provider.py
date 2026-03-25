"""Chronos model provider (stub).

Chronos is a family of pretrained time-series forecasting models from
Amazon.  This stub provides the correct structure so that the registry
can discover it; the actual inference logic will be added once the
``chronos-forecasting`` package is integrated.

Install with::

    pip install aio-sensor-intelligence[chronos]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import TASK_FORECAST, ModelProvider, ModelResult

logger = logging.getLogger(__name__)


class ChronosProvider(ModelProvider):
    """Stub provider for Amazon Chronos.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        ``"auto"`` to pick CUDA when available, or ``"cpu"``/``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-large",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device

    def load(self) -> None:
        logger.warning(
            "ChronosProvider.load() is a stub — install with: "
            "pip install aio-sensor-intelligence[chronos]"
        )

    def predict(self, data: np.ndarray, task: str, **kwargs: Any) -> ModelResult:
        raise NotImplementedError(
            "ChronosProvider is not yet implemented. "
            "Install with: pip install aio-sensor-intelligence[chronos]"
        )

    def supported_tasks(self) -> list[str]:
        return [TASK_FORECAST]

    def info(self) -> dict:
        return {
            "provider": "chronos",
            "model_name": self.model_name,
            "device": self.device,
            "supported_tasks": self.supported_tasks(),
            "status": "stub",
        }
