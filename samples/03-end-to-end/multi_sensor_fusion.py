"""
Multi-Sensor Fusion Engine
==========================

Reusable module for combining anomaly scores from multiple sensor channels
into a single fused assessment.  Designed for manufacturing process monitoring
where correlated deviations across sensors are more significant than
individual channel spikes.

Supported fusion strategies:
    - **weighted_average** — weighted mean of per-channel scores (default)
    - **max** — worst-case score across channels
    - **voting** — fraction of channels exceeding their individual thresholds

Usage::

    from multi_sensor_fusion import SensorFusionEngine

    engine = SensorFusionEngine(strategy="weighted_average")
    engine.add_channel("temperature", weight=1.2, threshold=0.65)
    engine.add_channel("pressure",    weight=1.0, threshold=0.70)
    engine.add_channel("vibration",   weight=0.8, threshold=0.60)

    engine.update("temperature", 0.45)
    engine.update("pressure",    0.80)
    engine.update("vibration",   0.30)

    print(engine.fused_score())   # weighted combination
    print(engine.is_anomaly())    # True / False
    print(engine.get_status())    # per-channel + fused summary
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _ChannelState:
    """Internal state for a single sensor channel."""

    name: str
    weight: float = 1.0
    threshold: float = 0.7
    latest_score: float = 0.0
    updated_at: float = 0.0  # time.monotonic() timestamp
    history: list[float] = field(default_factory=list)
    max_history: int = 100


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

FusionStrategy = Literal["weighted_average", "max", "voting"]


class SensorFusionEngine:
    """Combine anomaly scores from multiple sensor channels.

    Args:
        strategy: Fusion method — ``"weighted_average"``, ``"max"``, or
            ``"voting"``.
        global_threshold: Default anomaly threshold applied when calling
            :meth:`is_anomaly` without an explicit value.
        keep_history: Number of past scores to retain per channel (used
            for trend analysis and diagnostics).
    """

    VALID_STRATEGIES: tuple[str, ...] = ("weighted_average", "max", "voting")

    def __init__(
        self,
        strategy: FusionStrategy = "weighted_average",
        global_threshold: float = 0.7,
        keep_history: int = 100,
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from {self.VALID_STRATEGIES}"
            )
        self._strategy = strategy
        self._global_threshold = global_threshold
        self._keep_history = keep_history
        self._channels: dict[str, _ChannelState] = {}

    # -- channel management --------------------------------------------------

    def add_channel(
        self,
        name: str,
        weight: float = 1.0,
        threshold: float | None = None,
    ) -> None:
        """Register a sensor channel.

        Args:
            name: Unique channel identifier (e.g. ``"temperature"``).
            weight: Relative importance for weighted fusion.  Higher values
                give this channel more influence on the fused score.
            threshold: Per-channel anomaly threshold.  Falls back to the
                engine's *global_threshold* when ``None``.
        """
        if name in self._channels:
            raise ValueError(f"Channel {name!r} already registered")
        self._channels[name] = _ChannelState(
            name=name,
            weight=weight,
            threshold=threshold if threshold is not None else self._global_threshold,
            max_history=self._keep_history,
        )

    def remove_channel(self, name: str) -> None:
        """Remove a previously registered channel."""
        if name not in self._channels:
            raise KeyError(f"Channel {name!r} not found")
        del self._channels[name]

    @property
    def channel_names(self) -> list[str]:
        """Return names of all registered channels."""
        return list(self._channels)

    # -- score updates -------------------------------------------------------

    def update(self, channel: str, anomaly_score: float) -> None:
        """Update the latest anomaly score for *channel*.

        Args:
            channel: Name previously registered via :meth:`add_channel`.
            anomaly_score: Anomaly score in [0, 1].
        """
        if channel not in self._channels:
            raise KeyError(
                f"Unknown channel {channel!r}. "
                f"Registered: {list(self._channels)}"
            )
        ch = self._channels[channel]
        ch.latest_score = float(anomaly_score)
        ch.updated_at = time.monotonic()
        ch.history.append(ch.latest_score)
        if len(ch.history) > ch.max_history:
            ch.history = ch.history[-ch.max_history :]

    def update_batch(self, scores: dict[str, float]) -> None:
        """Update multiple channels at once.

        Args:
            scores: Mapping of channel name → anomaly score.
        """
        for channel, score in scores.items():
            self.update(channel, score)

    # -- fusion computation --------------------------------------------------

    def fused_score(self) -> float:
        """Compute the fused anomaly score across all channels.

        Returns:
            Combined score in [0, 1] (approximately — the ``max`` strategy
            and ``voting`` strategy guarantee this; weighted average does if
            individual scores are in [0, 1]).
        """
        if not self._channels:
            return 0.0
        return self._compute_fused()

    def is_anomaly(self, threshold: float | None = None) -> bool:
        """Return ``True`` if the fused score exceeds the threshold.

        Args:
            threshold: Override for the engine's global threshold.
        """
        thr = threshold if threshold is not None else self._global_threshold
        return self.fused_score() > thr

    # -- status / diagnostics ------------------------------------------------

    def get_status(self) -> dict:
        """Return a detailed status dictionary.

        Includes per-channel scores, whether each channel individually
        exceeds its threshold, the fused score, and the overall anomaly
        flag.

        Returns:
            Dictionary with keys ``channels``, ``fused_score``,
            ``is_anomaly``, ``strategy``, and ``global_threshold``.
        """
        channels_info: list[dict] = []
        for ch in self._channels.values():
            channels_info.append(
                {
                    "name": ch.name,
                    "weight": ch.weight,
                    "threshold": ch.threshold,
                    "latest_score": ch.latest_score,
                    "is_anomaly": ch.latest_score > ch.threshold,
                    "history_len": len(ch.history),
                }
            )

        fused = self.fused_score()
        return {
            "strategy": self._strategy,
            "global_threshold": self._global_threshold,
            "fused_score": fused,
            "is_anomaly": fused > self._global_threshold,
            "channels": channels_info,
        }

    def get_channel_history(self, channel: str) -> list[float]:
        """Return the score history for a specific channel."""
        if channel not in self._channels:
            raise KeyError(f"Unknown channel {channel!r}")
        return list(self._channels[channel].history)

    # -- strategy configuration ----------------------------------------------

    @property
    def strategy(self) -> str:
        """Currently active fusion strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: FusionStrategy) -> None:
        if value not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {value!r}. "
                f"Choose from {self.VALID_STRATEGIES}"
            )
        self._strategy = value

    @property
    def global_threshold(self) -> float:
        return self._global_threshold

    @global_threshold.setter
    def global_threshold(self, value: float) -> None:
        self._global_threshold = float(value)

    # -- internals -----------------------------------------------------------

    def _compute_fused(self) -> float:
        if self._strategy == "weighted_average":
            return self._fuse_weighted_average()
        if self._strategy == "max":
            return self._fuse_max()
        if self._strategy == "voting":
            return self._fuse_voting()
        raise ValueError(f"Unknown strategy: {self._strategy}")

    def _fuse_weighted_average(self) -> float:
        total_weight = sum(ch.weight for ch in self._channels.values())
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(
            ch.latest_score * ch.weight for ch in self._channels.values()
        )
        return weighted_sum / total_weight

    def _fuse_max(self) -> float:
        return max(ch.latest_score for ch in self._channels.values())

    def _fuse_voting(self) -> float:
        """Fraction of channels whose score exceeds their own threshold."""
        if not self._channels:
            return 0.0
        votes = sum(
            1 for ch in self._channels.values()
            if ch.latest_score > ch.threshold
        )
        return votes / len(self._channels)

    def __repr__(self) -> str:
        n = len(self._channels)
        return (
            f"SensorFusionEngine(strategy={self._strategy!r}, "
            f"channels={n}, threshold={self._global_threshold})"
        )
