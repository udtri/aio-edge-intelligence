"""Vibration sensor profile for rotating machinery (motor/pump).

Generates realistic vibration waveforms based on rotational dynamics.
The fundamental frequency and its harmonics model imbalance, misalignment,
and bearing defects commonly seen in industrial motors and pumps.
"""

import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Supported vibration anomaly injection types."""

    IMPACT = "impact"
    LOOSENESS = "looseness"
    GEAR_MESH_FAULT = "gear_mesh_fault"


class VibrationMotorProfile:
    """Simulates vibration sensor output for rotating machinery.

    The normal signal is a sum of sinusoids at the fundamental rotational
    frequency and its first few harmonics, plus Gaussian measurement noise.
    Degradation and anomaly modes modify this base signal to reproduce
    common fault signatures.

    Args:
        sampling_rate: Samples per second (Hz).
        fundamental_freq: Rotational frequency of the machine (Hz).
            Default 30 Hz ≈ 1800 RPM.
        noise_level: Standard deviation of additive Gaussian noise (g).
        n_harmonics: Number of harmonics above the fundamental to include.
    """

    def __init__(
        self,
        sampling_rate: float = 1000.0,
        fundamental_freq: float = 30.0,
        noise_level: float = 0.02,
        n_harmonics: int = 4,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.fundamental_freq = fundamental_freq
        self.noise_level = noise_level
        self.n_harmonics = n_harmonics

        # Degradation state (bearing wear)
        self._degradation_factor: float = 0.0  # 0 = healthy, 1 = severe
        self._degradation_rate: float = 0.001  # per generate() call

        # Pending one-shot anomaly
        self._pending_anomaly: AnomalyType | None = None

        # Internal RNG for reproducibility within a run
        self._rng = np.random.default_rng()

    # -- public API ----------------------------------------------------------

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate *n_samples* of vibration data (units: g).

        Each call advances the degradation model, so successive calls
        produce a gradually worsening signal when degradation is enabled.
        """
        t = np.arange(n_samples) / self.sampling_rate

        # Base harmonic signal
        signal = self._base_signal(t)

        # Layer degradation (growing harmonic amplitudes)
        signal += self._degradation_component(t)

        # Add measurement noise
        signal += self._rng.normal(0.0, self.noise_level, size=n_samples)

        # One-shot anomaly injection
        if self._pending_anomaly is not None:
            signal = self._apply_anomaly(signal, t)
            self._pending_anomaly = None

        # Advance degradation
        self._degradation_factor = min(1.0, self._degradation_factor + self._degradation_rate)

        return signal

    def inject_anomaly(self, anomaly_type: str) -> None:
        """Queue an anomaly to be injected in the next generate() call.

        Args:
            anomaly_type: One of 'impact', 'looseness', 'gear_mesh_fault'.
        """
        self._pending_anomaly = AnomalyType(anomaly_type)
        logger.info("Anomaly queued: %s", anomaly_type)

    def reset_degradation(self) -> None:
        """Reset the degradation model to a healthy state."""
        self._degradation_factor = 0.0

    # -- internal helpers ----------------------------------------------------

    def _base_signal(self, t: np.ndarray) -> np.ndarray:
        """Sum of fundamental + harmonics with 1/n amplitude roll-off."""
        signal = np.zeros_like(t)
        for n in range(1, self.n_harmonics + 2):  # 1x, 2x, … (n_harmonics+1)x
            freq = n * self.fundamental_freq
            amplitude = 1.0 / n  # natural roll-off
            phase = self._rng.uniform(0, 2 * np.pi)
            signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
        return signal

    def _degradation_component(self, t: np.ndarray) -> np.ndarray:
        """Simulates bearing wear: harmonics grow, sub-harmonics appear."""
        if self._degradation_factor < 1e-6:
            return np.zeros_like(t)

        component = np.zeros_like(t)
        # Bearing defect frequencies as non-integer multiples of fundamental
        defect_ratios = [3.56, 5.12, 7.24]  # typical BPFO-like ratios
        for i, ratio in enumerate(defect_ratios):
            freq = ratio * self.fundamental_freq
            amp = self._degradation_factor * 0.3 / (i + 1)
            component += amp * np.sin(2 * np.pi * freq * t)

        # Sub-harmonic at 0.5x (looseness precursor)
        component += (
            self._degradation_factor * 0.1
            * np.sin(2 * np.pi * 0.5 * self.fundamental_freq * t)
        )
        return component

    def _apply_anomaly(self, signal: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply the queued anomaly to the signal in-place."""
        n = len(signal)
        anomaly = self._pending_anomaly

        if anomaly == AnomalyType.IMPACT:
            # Sudden exponentially-decaying impact at a random location
            impact_idx = self._rng.integers(n // 4, 3 * n // 4)
            decay_samples = int(0.01 * self.sampling_rate)  # 10 ms ring-down
            amplitude = self._rng.uniform(5.0, 15.0)  # large spike
            for i in range(min(decay_samples, n - impact_idx)):
                signal[impact_idx + i] += amplitude * np.exp(-i / (decay_samples / 5))
            logger.debug("Impact anomaly at sample %d, amplitude %.1f g", impact_idx, amplitude)

        elif anomaly == AnomalyType.LOOSENESS:
            # Broadband noise increase across the full window
            broadband = self._rng.normal(0, 0.5, size=n)
            # Add truncated half-harmonics (mechanical looseness signature)
            for k in range(1, 10):
                freq = k * 0.5 * self.fundamental_freq
                broadband += 0.15 * np.sin(2 * np.pi * freq * t)
            signal += broadband

        elif anomaly == AnomalyType.GEAR_MESH_FAULT:
            # High-frequency burst modulated by rotation
            gear_teeth = 20
            mesh_freq = gear_teeth * self.fundamental_freq
            modulation = 0.5 * (1 + np.sin(2 * np.pi * self.fundamental_freq * t))
            burst = 2.0 * modulation * np.sin(2 * np.pi * mesh_freq * t)
            # Add sidebands (mesh_freq ± fundamental)
            burst += 0.8 * np.sin(2 * np.pi * (mesh_freq + self.fundamental_freq) * t)
            burst += 0.8 * np.sin(2 * np.pi * (mesh_freq - self.fundamental_freq) * t)
            signal += burst

        return signal
