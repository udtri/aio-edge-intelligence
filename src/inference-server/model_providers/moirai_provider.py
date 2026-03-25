"""MOIRAI model provider (stub).

MOIRAI is a masked-encoder-based universal forecasting transformer from
Salesforce.  This stub provides the correct structure so that the registry
can discover it; the actual inference logic will be added once the
``uni2ts`` package is integrated.

Install with::

    pip install aio-sensor-intelligence[moirai]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import TASK_FORECAST, ModelProvider, ModelResult

logger = logging.getLogger(__name__)


class MoiraiProvider(ModelProvider):
    """Stub provider for Salesforce MOIRAI.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        ``"auto"`` to pick CUDA when available, or ``"cpu"``/``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/moirai-1.1-R-large",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device

    def load(self) -> None:
        logger.warning(
            "MoiraiProvider.load() is a stub — install with: "
            "pip install aio-sensor-intelligence[moirai]"
        )

    def predict(self, data: np.ndarray, task: str, **kwargs: Any) -> ModelResult:
        raise NotImplementedError(
            "MoiraiProvider is not yet implemented. "
            "Install with: pip install aio-sensor-intelligence[moirai]"
        )

    def supported_tasks(self) -> list[str]:
        return [TASK_FORECAST]

    def info(self) -> dict:
        return {
            "provider": "moirai",
            "model_name": self.model_name,
            "device": self.device,
            "supported_tasks": self.supported_tasks(),
            "status": "stub",
        }
