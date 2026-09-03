"""Model-provider registry and factory for aio-sensor-intelligence.

Usage::

    from model_providers import get_provider

    provider = get_provider("timesfm", device="cuda")
    provider.load()
    result = provider.predict(data, task="forecasting")
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    ALL_TASKS,
    TASK_ANOMALY,
    TASK_CLASSIFY,
    TASK_FORECAST,
    TASK_IMPUTATION,
    ModelProvider,
    ModelResult,
)
from .custom_provider import CustomProvider

logger = logging.getLogger(__name__)

# Attempt to import providers that depend on optional packages.
# Each is guarded so that a missing dependency doesn't break the registry.

try:
    from .moment_provider import MomentProvider
except ImportError:
    MomentProvider = None  # type: ignore[assignment,misc]
    logger.warning(
        "momentfm is not installed — MomentProvider unavailable. "
        "Install with: pip install momentfm"
    )



# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type[ModelProvider]] = {
    "custom": CustomProvider,
}

if MomentProvider is not None:
    _REGISTRY["moment"] = MomentProvider

try:
    from .timesfm_provider import TimesFMProvider
except ImportError:
    TimesFMProvider = None  # type: ignore[assignment,misc]
    logger.warning(
        "timesfm is not installed — TimesFMProvider unavailable. "
        "Install with: pip install 'timesfm[torch]>=3.0.1,<4'"
    )

if TimesFMProvider is not None:
    _REGISTRY["timesfm"] = TimesFMProvider


def get_provider(name: str, **kwargs: Any) -> ModelProvider:
    """Instantiate a model provider by its registered name.

    Parameters
    ----------
    name : str
        Key in the provider registry (``"moment"``, ``"timesfm"``, or
        ``"custom"``).
    **kwargs
        Forwarded to the provider constructor (e.g. ``model_name``,
        ``device``).

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown provider '{name}'. Available: {available}"
        )
    logger.info("Creating provider '%s' with args %s", name, kwargs)
    return cls(**kwargs)


__all__ = [
    "ALL_TASKS",
    # Task constants
    "TASK_ANOMALY",
    "TASK_CLASSIFY",
    "TASK_FORECAST",
    "TASK_IMPUTATION",
    "CustomProvider",
    # Base types
    "ModelProvider",
    "ModelResult",
    # Concrete providers
    "MomentProvider",
    "TimesFMProvider",
    # Factory
    "get_provider",
]
