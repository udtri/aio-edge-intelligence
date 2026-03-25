"""Custom / bring-your-own model provider.

This module is a **template** that you can adapt to wrap your own model
behind the :class:`ModelProvider` interface.  Copy this file or edit it
in place — the only requirement is that you keep the four abstract
methods (``load``, ``predict``, ``supported_tasks``, ``info``).

Quick-start
-----------
1. Set ``model_path`` to the directory (or file) that contains your
   serialised model weights.
2. Implement ``load()`` to deserialise the weights into whatever runtime
   object your model needs (``torch.nn.Module``, ONNX session, etc.).
3. Implement ``predict()`` to convert the incoming numpy array into
   your model's expected input format, run inference, and wrap the
   output in a :class:`ModelResult`.
4. Update ``supported_tasks()`` to list the tasks your model handles.
5. Register it in ``__init__.py`` under a custom name, or just use
   ``"custom"`` as the provider key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import ModelProvider, ModelResult

logger = logging.getLogger(__name__)


class CustomProvider(ModelProvider):
    """Template provider for user-supplied models.

    Parameters
    ----------
    model_path : str
        Filesystem path to the model weights or directory.
    device : str
        ``"auto"`` to pick CUDA when available, or ``"cpu"``/``"cuda"``.
    """

    def __init__(
        self,
        model_path: str = "./models/custom",
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._model: Any = None

    # ------------------------------------------------------------------
    # ModelProvider interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights from ``self.model_path``.

        TODO: Replace the body of this method with your own loading
        logic.  For example::

            import torch
            self._model = torch.load(self.model_path / "model.pt",
                                     map_location=self.device)
            self._model.eval()
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {self.model_path}"
            )
        logger.info("CustomProvider: loading model from %s …", self.model_path)
        # TODO: load your model here
        self._model = None  # placeholder
        logger.info("CustomProvider: model loaded on %s", self.device)

    def predict(self, data: np.ndarray, task: str, **kwargs: Any) -> ModelResult:
        """Run inference on *data*.

        TODO: Replace the body with your own inference logic.  The
        incoming ``data`` array has shape ``[batch, channels, seq_len]``.
        Convert it to whatever format your model expects, run the
        forward pass, and return a :class:`ModelResult`.

        Example (PyTorch)::

            import torch
            tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self._model(tensor)
            return ModelResult(
                values=output.cpu().numpy(),
                task=task,
                metadata={"model_path": str(self.model_path)},
            )
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded — call load() before predict()."
            )
        # TODO: implement inference
        raise NotImplementedError(
            "CustomProvider.predict() is a template — "
            "replace this with your own inference logic."
        )

    def supported_tasks(self) -> list[str]:
        """Return the tasks your model supports.

        TODO: Update this list to match your model's capabilities.
        Use the constants from ``base.py`` (e.g. ``TASK_FORECAST``).
        """
        return []  # TODO: populate with your supported tasks

    def info(self) -> dict:
        return {
            "provider": "custom",
            "model_path": str(self.model_path),
            "device": self.device,
            "supported_tasks": self.supported_tasks(),
        }
