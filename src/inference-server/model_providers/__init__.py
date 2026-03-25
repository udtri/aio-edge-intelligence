"""Model-provider registry and factory for aio-sensor-intelligence.

Usage::

    from model_providers import get_provider

    provider = get_provider("moment", device="cuda")
    provider.load()
    result = provider.predict(data, task="forecasting")
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    ALL_TASKS,
    ModelProvider,
    ModelResult,
    TASK_ANOMALY,
    TASK_CLASSIFY,
    TASK_FORECAST,
    TASK_IMPUTATION,
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

try:
    from .moirai_provider import MoiraiProvider
except ImportError:
    MoiraiProvider = None  # type: ignore[assignment,misc]
    logger.warning(
        "uni2ts is not installed — MoiraiProvider unavailable. "
        "Install with: pip install uni2ts"
    )

try:
    from .chronos_provider import ChronosProvider
except ImportError:
    ChronosProvider = None  # type: ignore[assignment,misc]
    logger.warning(
        "chronos-forecasting is not installed — ChronosProvider unavailable. "
        "Install with: pip install chronos-forecasting"
    )

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type[ModelProvider]] = {
    "custom": CustomProvider,
}

if MomentProvider is not None:
    _REGISTRY["moment"] = MomentProvider
if MoiraiProvider is not None:
    _REGISTRY["moirai"] = MoiraiProvider
if ChronosProvider is not None:
    _REGISTRY["chronos"] = ChronosProvider


def get_provider(name: str, **kwargs: Any) -> ModelProvider:
    """Instantiate a model provider by its registered name.

    Parameters
    ----------
    name : str
        Key in the provider registry (``"moment"``, ``"moirai"``,
        ``"chronos"``, or ``"custom"``).
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
    # Factory
    "get_provider",
    # Base types
    "ModelProvider",
    "ModelResult",
    # Task constants
    "TASK_ANOMALY",
    "TASK_FORECAST",
    "TASK_CLASSIFY",
    "TASK_IMPUTATION",
    "ALL_TASKS",
    # Concrete providers
    "MomentProvider",
    "MoiraiProvider",
    "ChronosProvider",
    "CustomProvider",
]
