"""Temperature sensor profile for an industrial furnace.

Simulates realistic thermal cycling: heating ramp → hold → cooling,
including common fault modes such as runaway heating and cooling failure.
"""

import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Supported temperature anomaly injection types."""

    RUNAWAY_HEATING = "runaway_heating"
    COOLING_FAILURE = "cooling_failure"
    THERMAL_SHOCK = "thermal_shock"
    CALIBRATION_DRIFT = "calibration_drift"


class _CyclePhase(str, Enum):
    HEATING = "heating"
    HOLDING = "holding"
    COOLING = "cooling"


class TemperatureFurnaceProfile:
    """Simulates industrial furnace temperature readings (°C).

    The furnace cycles through heating → hold → cooling phases.  Each
    ``generate()`` call returns *n_samples* consecutive temperature values
    that seamlessly continue from the previous call.

    Args:
        sampling_rate: Samples per second.
        ambient_temp: Ambient / cold-start temperature (°C).
        target_temp: Setpoint during hold phase (°C).
        heating_rate: Temperature rise per second (°C/s).
        cooling_rate: Temperature drop per second (°C/s) — positive value.
        hold_duration_s: How long to hold at setpoint (seconds).
        noise_level: Gaussian sensor noise σ (°C).
    """

    def __init__(
        self,
        sampling_rate: float = 10.0,
        ambient_temp: float = 25.0,
        target_temp: float = 650.0,
        heating_rate: float = 8.0,
        cooling_rate: float = 3.0,
        hold_duration_s: float = 120.0,
        noise_level: float = 1.5,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.ambient_temp = ambient_temp
        self.target_temp = target_temp
        self.heating_rate = heating_rate
        self.cooling_rate = cooling_rate
        self.hold_duration_s = hold_duration_s
        self.noise_level = noise_level

        # State machine
        self._phase = _CyclePhase.HEATING
        self._current_temp = ambient_temp
        self._hold_elapsed = 0.0  # seconds spent in hold phase

        # Anomaly state
        self._pending_anomaly: AnomalyType | None = None
        self._drift_offset: float = 0.0  # cumulative calibration drift
        self._drift_rate: float = 0.0

        self._rng = np.random.default_rng()

    # -- public API ----------------------------------------------------------

    def generate(self, n_samples: int) -> np.ndarray:
        """Return *n_samples* consecutive temperature readings (°C)."""
        dt = 1.0 / self.sampling_rate
        temps = np.empty(n_samples)

        for i in range(n_samples):
            self._step(dt)
            temps[i] = self._current_temp

        # Sensor noise
        temps += self._rng.normal(0.0, self.noise_level, size=n_samples)

        # Apply calibration drift (additive offset)
        if self._drift_rate != 0.0:
            drift_array = self._drift_offset + self._drift_rate * np.arange(n_samples) * dt
            temps += drift_array
            self._drift_offset += self._drift_rate * n_samples * dt

        # One-shot anomalies
        if self._pending_anomaly is not None:
            temps = self._apply_anomaly(temps, dt)
            self._pending_anomaly = None

        return temps

    def inject_anomaly(self, anomaly_type: str) -> None:
        """Queue an anomaly for the next generate() call.

        Args:
            anomaly_type: One of 'runaway_heating', 'cooling_failure',
                'thermal_shock', 'calibration_drift'.
        """
        atype = AnomalyType(anomaly_type)

        if atype == AnomalyType.CALIBRATION_DRIFT:
            # Calibration drift is persistent — start a slow offset ramp
            self._drift_rate = self._rng.uniform(0.02, 0.1)  # °C/s
            logger.info("Calibration drift enabled: %.3f °C/s", self._drift_rate)
        else:
            self._pending_anomaly = atype
            logger.info("Anomaly queued: %s", anomaly_type)

    # -- state machine -------------------------------------------------------

    def _step(self, dt: float) -> None:
        """Advance the furnace model by *dt* seconds."""
        if self._phase == _CyclePhase.HEATING:
            self._current_temp += self.heating_rate * dt
            if self._current_temp >= self.target_temp:
                self._current_temp = self.target_temp
                self._phase = _CyclePhase.HOLDING
                self._hold_elapsed = 0.0

        elif self._phase == _CyclePhase.HOLDING:
            self._hold_elapsed += dt
            # Small oscillation around setpoint (PID-like behavior)
            self._current_temp = self.target_temp + 2.0 * np.sin(
                2 * np.pi * 0.05 * self._hold_elapsed
            )
            if self._hold_elapsed >= self.hold_duration_s:
                self._phase = _CyclePhase.COOLING

        elif self._phase == _CyclePhase.COOLING:
            # Exponential cooling towards ambient (Newton's law)
            tau = (self.target_temp - self.ambient_temp) / self.cooling_rate
            self._current_temp = self.ambient_temp + (
                self._current_temp - self.ambient_temp
            ) * np.exp(-dt / tau)
            if self._current_temp <= self.ambient_temp + 5.0:
                self._current_temp = self.ambient_temp
                self._phase = _CyclePhase.HEATING

    # -- anomaly logic -------------------------------------------------------

    def _apply_anomaly(self, temps: np.ndarray, dt: float) -> np.ndarray:
        n = len(temps)
        anomaly = self._pending_anomaly

        if anomaly == AnomalyType.RUNAWAY_HEATING:
            # Temperature keeps climbing past setpoint
            start_idx = n // 4
            for i in range(start_idx, n):
                ramp = self.heating_rate * 1.5 * (i - start_idx) * dt
                temps[i] += ramp
            logger.debug("Runaway heating applied from sample %d", start_idx)

        elif anomaly == AnomalyType.COOLING_FAILURE:
            # Temperature stays near peak — flatten the cooling curve
            peak_temp = np.max(temps)
            for i in range(n):
                if temps[i] < peak_temp - 20:
                    temps[i] = peak_temp - self._rng.uniform(0, 10)

        elif anomaly == AnomalyType.THERMAL_SHOCK:
            # Sudden large drop then partial recovery
            if n <= 1:
                temps[0] -= self._rng.uniform(150, 300)
            else:
                lo = max(1, n // 4)
                hi = max(lo + 1, 3 * n // 4)
                shock_idx = self._rng.integers(lo, hi)
                drop_magnitude = self._rng.uniform(150, 300)
                recovery_samples = int(2.0 * self.sampling_rate)
                for i in range(min(recovery_samples, n - shock_idx)):
                    frac = i / recovery_samples
                    temps[shock_idx + i] -= drop_magnitude * (1 - frac)
                logger.debug(
                    "Thermal shock at sample %d, drop %.0f °C",
                    shock_idx,
                    drop_magnitude,
                )

        return temps
