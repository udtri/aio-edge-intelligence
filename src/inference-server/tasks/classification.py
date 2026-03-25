"""Classification task — wraps a ModelProvider for sensor-signal fault classification.

.. note::

   MOMENT's classification head requires **fine-tuning** on labelled data
   before it can produce meaningful fault labels.  Out of the box the model
   will return raw logits; the label mapping below assumes a fine-tuned
   checkpoint trained on industrial vibration data.  See
   ``notebooks/fine_tune_classification.ipynb`` (TODO) for a training
   walkthrough.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from ..model_providers.base import TASK_CLASSIFY, ModelProvider, ModelResult
from ..schemas import ClassificationResult

logger = logging.getLogger(__name__)

# Predefined fault labels for rotating-machinery vibration analysis.
# The index in this list matches the model's output class index after
# fine-tuning on the target label set.
DEFAULT_FAULT_LABELS: list[str] = [
    "normal",
    "bearing_fault",
    "imbalance",
    "misalignment",
    "looseness",
]


class Classifier:
    """High-level sensor-data classifier built on a :class:`ModelProvider`.

    Parameters
    ----------
    provider : ModelProvider
        A loaded model provider that supports ``TASK_CLASSIFY``.
    labels : list[str] or None
        Ordered class labels.  Defaults to :data:`DEFAULT_FAULT_LABELS`.
    window_size : int
        Expected input window length (default 512).

    Notes
    -----
    MOMENT's classification head outputs raw logits by default.  For
    meaningful results you **must** fine-tune the model on a labelled
    dataset that matches *labels*.  Without fine-tuning, the predicted
    probabilities and labels will be unreliable.
    """

    def __init__(
        self,
        provider: ModelProvider,
        labels: list[str] | None = None,
        window_size: int = 512,
    ) -> None:
        if TASK_CLASSIFY not in provider.supported_tasks():
            raise ValueError(
                f"Provider does not support '{TASK_CLASSIFY}'. "
                f"Supported: {provider.supported_tasks()}"
            )
        self.provider = provider
        self.labels = labels or list(DEFAULT_FAULT_LABELS)
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        data: np.ndarray,
        sensor_id: str | None = None,
    ) -> ClassificationResult:
        """Classify a sensor-data window.

        Parameters
        ----------
        data : np.ndarray
            Input array.  Shape ``(seq_len,)`` or ``(batch, channels, seq_len)``.
        sensor_id : str or None
            Optional sensor identifier.

        Returns
        -------
        ClassificationResult
        """
        data = self._prepare_input(data)
        result: ModelResult = self.provider.predict(data, task=TASK_CLASSIFY)
        logits = result.values.flatten()

        probabilities = self._softmax(logits)
        label_probs = self._build_label_probs(probabilities)

        top_idx = int(np.argmax(probabilities))
        top_label = self.labels[top_idx] if top_idx < len(self.labels) else f"class_{top_idx}"
        confidence = float(probabilities[top_idx])

        return ClassificationResult(
            sensor_id=sensor_id,
            label=top_label,
            confidence=confidence,
            probabilities=label_probs,
            timestamp=datetime.utcnow(),
        )

    def classify_top_k(
        self,
        data: np.ndarray,
        k: int = 3,
        sensor_id: str | None = None,
    ) -> list[ClassificationResult]:
        """Return the top-*k* classification results, sorted by confidence.

        Parameters
        ----------
        data : np.ndarray
            Input sensor window.
        k : int
            Number of top predictions to return.
        sensor_id : str or None
            Optional sensor identifier.

        Returns
        -------
        list[ClassificationResult]
        """
        data = self._prepare_input(data)
        result: ModelResult = self.provider.predict(data, task=TASK_CLASSIFY)
        logits = result.values.flatten()

        probabilities = self._softmax(logits)
        label_probs = self._build_label_probs(probabilities)
        sorted_indices = np.argsort(probabilities)[::-1][:k]

        results: list[ClassificationResult] = []
        for idx in sorted_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append(
                ClassificationResult(
                    sensor_id=sensor_id,
                    label=label,
                    confidence=float(probabilities[idx]),
                    probabilities=label_probs,
                    timestamp=datetime.utcnow(),
                )
            )
        return results

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

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically-stable softmax."""
        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    def _build_label_probs(self, probabilities: np.ndarray) -> dict[str, float]:
        """Zip labels with their probability values."""
        probs: dict[str, float] = {}
        for i, p in enumerate(probabilities):
            label = self.labels[i] if i < len(self.labels) else f"class_{i}"
            probs[label] = round(float(p), 6)
        return probs
