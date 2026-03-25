"""Task pipeline modules for the inference server.

Each task wraps a :class:`~model_providers.base.ModelProvider` and adds
domain-specific logic (thresholding, RUL estimation, severity levels, etc.).
"""

from .anomaly_detection import AnomalyDetector, RollingStats, Severity
from .classification import Classifier, DEFAULT_FAULT_LABELS
from .forecasting import Forecaster

__all__ = [
    "AnomalyDetector",
    "Classifier",
    "DEFAULT_FAULT_LABELS",
    "Forecaster",
    "RollingStats",
    "Severity",
]
