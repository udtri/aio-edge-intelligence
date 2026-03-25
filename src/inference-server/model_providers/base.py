"""Abstract base class and shared types for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Task name constants
# ---------------------------------------------------------------------------
TASK_ANOMALY: str = "anomaly_detection"
TASK_FORECAST: str = "forecasting"
TASK_CLASSIFY: str = "classification"
TASK_IMPUTATION: str = "imputation"

ALL_TASKS: list[str] = [TASK_ANOMALY, TASK_FORECAST, TASK_CLASSIFY, TASK_IMPUTATION]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    """Container for inference results returned by any provider."""

    values: np.ndarray
    task: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------
class ModelProvider(ABC):
    """Interface that every model provider must implement."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialise any runtime state."""

    @abstractmethod
    def predict(self, data: np.ndarray, task: str, **kwargs) -> ModelResult:
        """Run inference on *data* for the given *task*.

        Parameters
        ----------
        data : np.ndarray
            Input array with shape ``[batch, channels, seq_len]``.
        task : str
            One of the ``TASK_*`` constants defined in this module.
        **kwargs
            Provider-specific options (forecast horizon, threshold, …).

        Returns
        -------
        ModelResult
        """

    @abstractmethod
    def supported_tasks(self) -> list[str]:
        """Return the list of task names this provider can handle."""

    @abstractmethod
    def info(self) -> dict:
        """Return model metadata (name, version, parameter count, etc.)."""
