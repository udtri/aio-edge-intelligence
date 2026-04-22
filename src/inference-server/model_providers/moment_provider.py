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
        default_task: str = TASK_ANOMALY,
        forecast_horizon: int = 96,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._default_task = default_task
        self._forecast_horizon = forecast_horizon
        # Keyed by MOMENT task name so we can cache pipelines per-task.
        self._pipelines: dict[str, Any] = {}
        self._current_task: str | None = None

    # ------------------------------------------------------------------
    # ModelProvider interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Pre-load the default pipeline."""
        if MOMENTPipeline is None:
            raise ImportError(
                "momentfm is not installed. "
                "Install with: pip install momentfm"
            )
        self._get_pipeline(self._default_task)
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

        # MOMENT forward() uses keyword-only args: (*, x_enc, input_mask, mask)
        batch_size, _n_channels, seq_len = tensor.shape
        input_mask = torch.ones(batch_size, seq_len, device=self.device)

        moment_task = _MOMENT_TASK_MAP[task]

        with torch.no_grad():
            # The pretrained model's task_name may differ from what we want,
            # so call the specific task method directly.
            if moment_task == "reconstruction":
                output = pipeline.reconstruction(
                    x_enc=tensor, input_mask=input_mask, **kwargs
                )
                result_np = output.reconstruction.cpu().numpy()
            elif moment_task == "forecasting":
                # Ensure task_name is set so forward routing works
                pipeline.task_name = "forecasting"
                output = pipeline(x_enc=tensor, input_mask=input_mask, **kwargs)
                # Forecast output stored in .forecast; shape [B, C, seq_len]
                result_np = output.forecast.cpu().numpy()
            elif moment_task == "classification":
                output = pipeline.classify(
                    x_enc=tensor, input_mask=input_mask, **kwargs
                )
                result_np = output.logits.cpu().numpy()
            elif moment_task == "imputation":
                # Imputation needs a mask indicating missing values
                mask = kwargs.pop("mask", None)
                if mask is not None:
                    mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
                output = pipeline.reconstruction(
                    x_enc=tensor, input_mask=input_mask, mask=mask, **kwargs
                )
                result_np = output.reconstruction.cpu().numpy()
            else:
                # Fallback: call via forward()
                output = pipeline(x_enc=tensor, input_mask=input_mask, **kwargs)
                # Try common output attributes
                for attr in ("reconstruction", "forecast", "logits", "embeddings"):
                    val = getattr(output, attr, None)
                    if val is not None:
                        result_np = val.cpu().numpy()
                        break
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
        """Return (and cache) a MOMENTPipeline for *task*.

        Note: MOMENT-1-large from HuggingFace always loads with task_name
        ``'reconstruction'`` and a ``PretrainHead``.  All tasks share the
        same encoder; the task-specific routing happens inside the named
        methods (``reconstruction()``, ``forecast()``, etc.).  We therefore
        cache a **single** pipeline and set ``task_name`` dynamically in
        :meth:`predict`.
        """
        moment_task = _MOMENT_TASK_MAP.get(task)
        if moment_task is None:
            raise ValueError(
                f"Unsupported task '{task}'. Choose from {list(_MOMENT_TASK_MAP)}"
            )

        # Single shared pipeline — all tasks use the same pretrained weights.
        if "shared" not in self._pipelines:
            logger.info("Building MOMENT pipeline …")
            model_kwargs: dict[str, Any] = {
                "task_name": "reconstruction",
                "forecast_horizon": self._forecast_horizon,
            }
            pipeline = MOMENTPipeline.from_pretrained(
                self.model_name,
                model_kwargs=model_kwargs,
            )
            pipeline.init()
            if self.device != "cpu":
                pipeline = pipeline.to(self.device)
            self._pipelines["shared"] = pipeline

        self._current_task = moment_task
        return self._pipelines["shared"]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
