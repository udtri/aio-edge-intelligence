"""Pressure sensor profile for a hydraulic system.

Simulates realistic hydraulic pressure readings including steady-state
operation, work cycles (actuator movements), and common fault modes.
"""

import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Supported pressure anomaly injection types."""

    LEAK = "leak"
    BLOCKAGE = "blockage"
    CAVITATION = "cavitation"


class _SystemState(str, Enum):
    IDLE = "idle"
    ACTUATING = "actuating"
    RETURNING = "returning"


class PressureHydraulicProfile:
    """Simulates hydraulic system pressure readings (bar).

    The system alternates between idle (steady-state around *setpoint*)
    and work cycles where an actuator extends and retracts, causing
    characteristic pressure ramps and dips.

    Args:
        sampling_rate: Samples per second.
        setpoint: Nominal system pressure (bar).
        noise_level: Gaussian sensor noise σ (bar).
        cycle_probability: Probability per sample of starting a work cycle
            while idle.  Keeps the signal interesting without being constant.
        actuator_pressure_delta: Additional pressure during actuation (bar).
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        setpoint: float = 150.0,
        noise_level: float = 0.8,
        cycle_probability: float = 0.002,
        actuator_pressure_delta: float = 40.0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.setpoint = setpoint
        self.noise_level = noise_level
        self.cycle_probability = cycle_probability
        self.actuator_pressure_delta = actuator_pressure_delta

        # State
        self._state = _SystemState.IDLE
        self._current_pressure = setpoint
        self._cycle_progress: float = 0.0  # 0→1 through actuation/return
        self._cycle_duration_samples: int = 0

        # Anomaly
        self._pending_anomaly: AnomalyType | None = None
        self._leak_active: bool = False
        self._leak_rate: float = 0.0  # bar/sample

        self._rng = np.random.default_rng()

    # -- public API ----------------------------------------------------------

    def generate(self, n_samples: int) -> np.ndarray:
        """Return *n_samples* consecutive pressure readings (bar)."""
        pressures = np.empty(n_samples)

        for i in range(n_samples):
            self._step()
            pressures[i] = self._current_pressure

        # Sensor noise
        pressures += self._rng.normal(0.0, self.noise_level, size=n_samples)

        # Ongoing leak (gradual decline)
        if self._leak_active:
            leak_curve = self._leak_rate * np.arange(n_samples)
            pressures -= leak_curve
            self._current_pressure -= self._leak_rate * n_samples
            self._current_pressure = max(self._current_pressure, 0.0)

        # One-shot anomalies
        if self._pending_anomaly is not None:
            pressures = self._apply_anomaly(pressures)
            self._pending_anomaly = None

        return pressures

    def inject_anomaly(self, anomaly_type: str) -> None:
        """Queue or activate an anomaly.

        Args:
            anomaly_type: One of 'leak', 'blockage', 'cavitation'.
        """
        atype = AnomalyType(anomaly_type)

        if atype == AnomalyType.LEAK:
            self._leak_active = True
            self._leak_rate = self._rng.uniform(0.01, 0.05)
            logger.info("Hydraulic leak activated: %.3f bar/sample", self._leak_rate)
        else:
            self._pending_anomaly = atype
            logger.info("Anomaly queued: %s", anomaly_type)

    # -- state machine -------------------------------------------------------

    def _step(self) -> None:
        """Advance the hydraulic model by one sample."""
        if self._state == _SystemState.IDLE:
            # Small pump ripple at idle
            ripple_freq = 12.0  # Hz, typical gear-pump ripple
            t = 1.0 / self.sampling_rate
            self._current_pressure = self.setpoint + 1.5 * np.sin(
                2 * np.pi * ripple_freq * self._rng.uniform(0, 1)
            )
            # Randomly start a work cycle
            if self._rng.random() < self.cycle_probability:
                self._state = _SystemState.ACTUATING
                self._cycle_progress = 0.0
                # Duration: 0.5–2 seconds
                duration_s = self._rng.uniform(0.5, 2.0)
                self._cycle_duration_samples = max(
                    int(duration_s * self.sampling_rate), 1
                )

        elif self._state == _SystemState.ACTUATING:
            # Smooth pressure ramp up (half-sine envelope)
            frac = self._cycle_progress / self._cycle_duration_samples
            self._current_pressure = self.setpoint + self.actuator_pressure_delta * np.sin(
                np.pi * frac / 2
            )
            self._cycle_progress += 1
            if self._cycle_progress >= self._cycle_duration_samples:
                self._state = _SystemState.RETURNING
                self._cycle_progress = 0.0

        elif self._state == _SystemState.RETURNING:
            # Pressure returns to setpoint (mirror of actuation)
            frac = self._cycle_progress / self._cycle_duration_samples
            self._current_pressure = self.setpoint + self.actuator_pressure_delta * np.cos(
                np.pi * frac / 2
            )
            self._cycle_progress += 1
            if self._cycle_progress >= self._cycle_duration_samples:
                self._state = _SystemState.IDLE

    # -- anomaly logic -------------------------------------------------------

    def _apply_anomaly(self, pressures: np.ndarray) -> np.ndarray:
        n = len(pressures)
        anomaly = self._pending_anomaly

        if anomaly == AnomalyType.BLOCKAGE:
            # Sudden pressure spike from a downstream blockage
            spike_idx = self._rng.integers(n // 4, 3 * n // 4)
            spike_mag = self._rng.uniform(80, 160)
            decay = int(0.5 * self.sampling_rate)
            for i in range(min(decay, n - spike_idx)):
                frac = i / decay
                pressures[spike_idx + i] += spike_mag * np.exp(-3 * frac)
            logger.debug("Blockage spike at sample %d, +%.0f bar", spike_idx, spike_mag)

        elif anomaly == AnomalyType.CAVITATION:
            # Rapid oscillations with pressure dips below setpoint
            start = self._rng.integers(0, n // 2)
            length = min(int(1.0 * self.sampling_rate), n - start)
            t = np.arange(length) / self.sampling_rate
            # Multi-frequency oscillation
            osc = (
                15 * np.sin(2 * np.pi * 120 * t)
                + 8 * np.sin(2 * np.pi * 240 * t)
                + 5 * self._rng.normal(size=length)
            )
            pressures[start : start + length] += osc
            # Cavitation causes momentary vacuum — clamp negative
            pressures = np.maximum(pressures, 0.0)
            logger.debug("Cavitation event from sample %d, length %d", start, length)

        return pressures
