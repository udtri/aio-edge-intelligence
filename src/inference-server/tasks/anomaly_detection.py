"""Anomaly detection task — wraps a ModelProvider for real-time and batch anomaly scoring.

Supports both one-shot ``detect()`` calls on a NumPy array and streaming
``detect_stream()`` calls backed by a sliding-window buffer.  Thresholds
are auto-derived from model output statistics unless overridden.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from model_providers.base import TASK_ANOMALY, ModelProvider, ModelResult
from schemas import AnomalyResult

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Anomaly severity levels aligned with predictive-maintenance guidance."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

# Default score boundaries for severity classification.
_SEVERITY_THRESHOLDS: dict[Severity, tuple[float, float]] = {
    Severity.NORMAL: (0.0, 0.5),
    Severity.WARNING: (0.5, 0.7),
    Severity.CRITICAL: (0.7, 1.0),
}


@dataclass
class RollingStats:
    """Maintains exponentially-weighted rolling statistics for adaptive thresholding."""

    mean: float = 0.0
    var: float = 0.0
    count: int = 0
    alpha: float = 0.05  # smoothing factor

    def update(self, value: float) -> None:
        """Update running mean/variance with a new observation."""
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.var = 0.0
            return
        delta = value - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 0.0)))

    @property
    def adaptive_threshold(self) -> float:
        """Return mean + 2*std as an adaptive anomaly threshold."""
        return self.mean + 2.0 * self.std


class AnomalyDetector:
    """High-level anomaly detection built on top of a :class:`ModelProvider`.

    Parameters
    ----------
    provider : ModelProvider
        A loaded model provider that supports ``TASK_ANOMALY``.
    threshold : float or None
        Fixed anomaly threshold.  When *None* (default) the threshold is
        derived automatically from model output statistics (mean + 2·std).
    window_size : int
        Expected input window length (default 512 for MOMENT).
    severity_thresholds : dict or None
        Custom ``{Severity: (lo, hi)}`` mapping.  Defaults to module-level
        ``_SEVERITY_THRESHOLDS``.
    """

    def __init__(
        self,
        provider: ModelProvider,
        threshold: float | None = None,
        window_size: int = 512,
        severity_thresholds: dict[Severity, tuple[float, float]] | None = None,
    ) -> None:
        if TASK_ANOMALY not in provider.supported_tasks():
            raise ValueError(
                f"Provider does not support '{TASK_ANOMALY}'. "
                f"Supported: {provider.supported_tasks()}"
            )
        self.provider = provider
        self._fixed_threshold = threshold
        self.window_size = window_size
        self.severity_thresholds = severity_thresholds or dict(_SEVERITY_THRESHOLDS)

        # Adaptive stats tracker
        self._rolling = RollingStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        data: np.ndarray,
        threshold: float | None = None,
        sensor_id: str | None = None,
    ) -> AnomalyResult:
        """Run anomaly detection on a batch of data.

        Parameters
        ----------
        data : np.ndarray
            Input array.  Accepted shapes:
            - ``(seq_len,)`` — single univariate window (auto-expanded).
            - ``(batch, channels, seq_len)`` — raw provider format.
        threshold : float or None
            Override threshold for this call.  Falls back to the instance
            threshold or auto-derived value.
        sensor_id : str or None
            Optional sensor identifier attached to the result.

        Returns
        -------
        AnomalyResult
        """
        data = self._prepare_input(data)
        result: ModelResult = self.provider.predict(data, task=TASK_ANOMALY)
        scores = self._compute_scores(data, result)
        effective_threshold = self._resolve_threshold(scores, threshold)

        # Update adaptive stats with the mean score of this window
        self._rolling.update(float(np.mean(scores)))

        is_anomaly = bool(np.max(scores) >= effective_threshold)

        return AnomalyResult(
            sensor_id=sensor_id,
            anomaly_scores=scores.tolist(),
            threshold=effective_threshold,
            is_anomaly=is_anomaly,
            severity=self.classify_severity(float(np.max(scores))),
            timestamp=datetime.utcnow(),
        )

    def detect_stream(
        self,
        window_buffer: deque,
        sensor_id: str | None = None,
    ) -> AnomalyResult:
        """Detect anomalies from a real-time sliding window buffer.

        Parameters
        ----------
        window_buffer : deque
            A deque of scalar samples.  Must contain at least
            ``self.window_size`` elements; only the last ``window_size``
            are used.
        sensor_id : str or None
            Optional sensor identifier.

        Returns
        -------
        AnomalyResult
        """
        if len(window_buffer) < self.window_size:
            raise ValueError(
                f"Buffer has {len(window_buffer)} samples, "
                f"need at least {self.window_size}"
            )
        # Take the most recent window
        window = np.array(list(window_buffer)[-self.window_size :], dtype=np.float64)
        return self.detect(window, sensor_id=sensor_id)

    def classify_severity(self, score: float) -> str:
        """Map an anomaly score to a severity label."""
        for severity, (lo, hi) in self.severity_thresholds.items():
            if lo <= score < hi:
                return severity.value
        # Score >= 1.0 → critical
        return Severity.CRITICAL.value

    def get_rolling_stats(self) -> dict:
        """Return current adaptive rolling statistics."""
        return {
            "mean": self._rolling.mean,
            "std": self._rolling.std,
            "count": self._rolling.count,
            "adaptive_threshold": self._rolling.adaptive_threshold,
        }

    def reset_rolling_stats(self) -> None:
        """Reset the adaptive statistics tracker."""
        self._rolling = RollingStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, data: np.ndarray) -> np.ndarray:
        """Reshape input to ``(batch, channels, seq_len)``."""
        if data.ndim == 1:
            return data.reshape(1, 1, -1)
        if data.ndim == 2:
            # Assume (channels, seq_len) → add batch dim
            return data[np.newaxis, ...]
        return data

    def _compute_scores(
        self, original: np.ndarray, result: ModelResult
    ) -> np.ndarray:
        """Derive per-timestep anomaly scores via reconstruction error.

        MOMENT's anomaly task returns a *reconstructed* signal.  The score
        is the point-wise absolute error, normalised to [0, 1] by the
        range of the original signal.
        """
        reconstructed = result.values
        # Ensure shapes match for element-wise subtraction
        if reconstructed.shape != original.shape:
            reconstructed = reconstructed.reshape(original.shape)

        error = np.abs(original - reconstructed)
        # Normalise by the range of the original window (avoid div-by-zero)
        signal_range = np.ptp(original)
        if signal_range < 1e-8:
            signal_range = 1.0
        scores = error / signal_range
        return scores.flatten()

    def _resolve_threshold(
        self, scores: np.ndarray, override: float | None
    ) -> float:
        """Pick the effective threshold in priority order:
        1. Per-call *override*
        2. Instance-level ``_fixed_threshold``
        3. Auto: mean + 2*std of the current score array
        """
        if override is not None:
            return override
        if self._fixed_threshold is not None:
            return self._fixed_threshold
        # Auto threshold from current scores
        return float(np.mean(scores) + 2.0 * np.std(scores))
