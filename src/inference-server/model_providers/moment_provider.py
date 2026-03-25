"""MOMENT model provider — default provider for aio-sensor-intelligence.

MOMENT is a family of foundation models for time-series that supports
anomaly detection, forecasting, classification, and imputation out of the
box.  This provider wraps ``momentfm.MOMENTPipeline`` and exposes it
through the unified :class:`ModelProvider` interface.
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
    from momentfm import MOMENTPipeline
except ImportError:  # pragma: no cover
    MOMENTPipeline = None  # type: ignore[assignment,misc]

from .base import (
    ALL_TASKS,
    ModelProvider,
    ModelResult,
    TASK_ANOMALY,
    TASK_CLASSIFY,
    TASK_FORECAST,
    TASK_IMPUTATION,
)

logger = logging.getLogger(__name__)

# Maps our canonical task names to the names MOMENT expects.
_MOMENT_TASK_MAP: dict[str, str] = {
    TASK_ANOMALY: "reconstruction",
    TASK_FORECAST: "forecasting",
    TASK_CLASSIFY: "classification",
    TASK_IMPUTATION: "imputation",
}


class MomentProvider(ModelProvider):
    """Provider backed by the MOMENT foundation model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default ``AutonLab/MOMENT-1-large``).
    device : str
        ``"auto"`` (default) to pick CUDA when available, or ``"cpu"``/``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "AutonLab/MOMENT-1-large",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        # Keyed by MOMENT task name so we can cache pipelines per-task.
        self._pipelines: dict[str, Any] = {}
        self._current_task: str | None = None

    # ------------------------------------------------------------------
    # ModelProvider interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Pre-load the default (forecasting) pipeline."""
        if MOMENTPipeline is None:
            raise ImportError(
                "momentfm is not installed. "
                "Install with: pip install momentfm"
            )
        self._get_pipeline(TASK_FORECAST)
        logger.info("MomentProvider loaded (%s) on %s", self.model_name, self.device)

    def predict(self, data: np.ndarray, task: str, **kwargs: Any) -> ModelResult:
        """Run inference with MOMENT.

        Parameters
        ----------
        data : np.ndarray
            Shape ``[batch, channels, seq_len]``.
        task : str
            One of the ``TASK_*`` constants.
        **kwargs
            Extra arguments forwarded to the MOMENT pipeline (e.g.
            ``forecast_horizon``).
        """
        if torch is None:
            raise ImportError("PyTorch is required but not installed.")

        pipeline = self._get_pipeline(task)
        tensor = torch.tensor(data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = pipeline(tensor, **kwargs)

        # MOMENT outputs vary by task; normalise to numpy.
        if hasattr(output, "output"):
            result_np = output.output.cpu().numpy()
        elif isinstance(output, torch.Tensor):
            result_np = output.cpu().numpy()
        else:
            result_np = np.asarray(output)

        return ModelResult(
            values=result_np,
            task=task,
            metadata={"model": self.model_name, "device": self.device},
        )

    def supported_tasks(self) -> list[str]:
        return list(ALL_TASKS)

    def info(self) -> dict:
        return {
            "provider": "moment",
            "model_name": self.model_name,
            "device": self.device,
            "supported_tasks": self.supported_tasks(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pipeline(self, task: str) -> Any:
        """Return (and cache) a MOMENTPipeline for *task*."""
        moment_task = _MOMENT_TASK_MAP.get(task)
        if moment_task is None:
            raise ValueError(
                f"Unsupported task '{task}'. Choose from {list(_MOMENT_TASK_MAP)}"
            )

        if moment_task not in self._pipelines:
            logger.info("Building MOMENT pipeline for task '%s' …", moment_task)
            pipeline = MOMENTPipeline.from_pretrained(
                self.model_name,
                model_kwargs={"task_name": moment_task},
            )
            pipeline.init()
            if self.device != "cpu":
                pipeline = pipeline.to(self.device)
            self._pipelines[moment_task] = pipeline

        self._current_task = moment_task
        return self._pipelines[moment_task]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
