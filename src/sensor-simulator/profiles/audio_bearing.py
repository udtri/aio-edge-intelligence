"""Acoustic emission profile for rolling-element bearings.

Simulates bearing vibration / acoustic data using physics-based fault
frequency calculations derived from bearing geometry.  As damage
progresses, characteristic defect frequencies emerge from the noise floor
and grow in amplitude.

Reference
---------
Bearing defect frequencies (for a stationary outer race):
    BPFO = (n/2) · f_r · (1 − d/D · cos α)
    BPFI = (n/2) · f_r · (1 + d/D · cos α)
    BSF  = (D/(2d)) · f_r · (1 − (d/D · cos α)²)
    FTF  = (1/2) · f_r · (1 − d/D · cos α)

where n = number of rolling elements, f_r = shaft rotation frequency,
d = ball diameter, D = pitch diameter, α = contact angle.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class FaultType(str, Enum):
    """Bearing fault locations."""

    OUTER_RACE = "outer_race"
    INNER_RACE = "inner_race"
    BALL = "ball"
    CAGE = "cage"


@dataclass
class BearingGeometry:
    """Physical dimensions of a deep-groove ball bearing (e.g., 6205).

    All lengths in mm; angles in degrees.
    """

    n_balls: int = 9
    ball_diameter: float = 7.94  # mm
    pitch_diameter: float = 38.5  # mm
    contact_angle: float = 0.0  # degrees (0 for deep-groove)

    # Derived fault frequencies (multiples of shaft speed)
    def fault_orders(self) -> dict[str, float]:
        """Return fault frequency *orders* (multiples of shaft freq)."""
        d, D = self.ball_diameter, self.pitch_diameter
        n = self.n_balls
        alpha = np.radians(self.contact_angle)
        ratio = d / D * np.cos(alpha)
        return {
            "BPFO": (n / 2) * (1 - ratio),
            "BPFI": (n / 2) * (1 + ratio),
            "BSF": (D / (2 * d)) * (1 - ratio ** 2),
            "FTF": 0.5 * (1 - ratio),
        }


@dataclass
class DamageState:
    """Tracks progressive damage for each fault type."""

    severity: dict[str, float] = field(default_factory=lambda: {
        "BPFO": 0.0,
        "BPFI": 0.0,
        "BSF": 0.0,
        "FTF": 0.0,
    })
    progression_rate: dict[str, float] = field(default_factory=lambda: {
        "BPFO": 0.0,
        "BPFI": 0.0,
        "BSF": 0.0,
        "FTF": 0.0,
    })


class AudioBearingProfile:
    """Simulates bearing acoustic emission data.

    In the healthy state the signal is low-amplitude broadband noise.
    When damage is introduced via ``inject_fault()``, the corresponding
    defect frequency and its harmonics gradually emerge from the noise
    floor, reproducing the classic bearing degradation signature.

    Args:
        sampling_rate: Samples per second (Hz).  Typical AE sensors run
            at 20–100 kHz; we default to 12 kHz for manageable data size.
        shaft_freq: Shaft rotational frequency (Hz).  25 Hz ≈ 1500 RPM.
        noise_floor: RMS amplitude of healthy broadband noise.
        geometry: Bearing geometry parameters.
    """

    def __init__(
        self,
        sampling_rate: float = 12000.0,
        shaft_freq: float = 25.0,
        noise_floor: float = 0.005,
        geometry: BearingGeometry | None = None,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.shaft_freq = shaft_freq
        self.noise_floor = noise_floor
        self.geometry = geometry or BearingGeometry()

        self._fault_orders = self.geometry.fault_orders()
        self._damage = DamageState()
        self._rng = np.random.default_rng()

        # Running phase accumulator for continuous signal
        self._phase_offset: float = 0.0

    # -- public API ----------------------------------------------------------

    def generate(self, n_samples: int) -> np.ndarray:
        """Return *n_samples* of acoustic emission data.

        The signal is the sum of:
        1. Broadband background noise (healthy baseline).
        2. For each active fault, impulse trains at the defect frequency
           modulated by an exponential ring-down envelope.
        """
        t = (self._phase_offset + np.arange(n_samples)) / self.sampling_rate

        # 1. Broadband noise
        signal = self._rng.normal(0, self.noise_floor, size=n_samples)

        # 2. Fault components
        for label, order in self._fault_orders.items():
            severity = self._damage.severity[label]
            if severity < 1e-6:
                continue
            signal += self._fault_component(t, order, severity, label)

        # Advance phase
        self._phase_offset += n_samples

        # Progress damage
        self._advance_damage()

        return signal

    def inject_fault(self, fault_type: str, rate: float = 0.005) -> None:
        """Begin progressive damage at a specific fault location.

        Args:
            fault_type: One of 'outer_race', 'inner_race', 'ball', 'cage'.
            rate: Severity increase per ``generate()`` call (0→1 scale).
        """
        mapping = {
            FaultType.OUTER_RACE: "BPFO",
            FaultType.INNER_RACE: "BPFI",
            FaultType.BALL: "BSF",
            FaultType.CAGE: "FTF",
        }
        key = mapping[FaultType(fault_type)]
        self._damage.progression_rate[key] = rate
        logger.info("Fault injected: %s (%s), rate=%.4f", fault_type, key, rate)

    def reset(self) -> None:
        """Reset all damage to healthy state."""
        self._damage = DamageState()
        self._phase_offset = 0.0

    # -- internal helpers ----------------------------------------------------

    def _fault_component(
        self,
        t: np.ndarray,
        order: float,
        severity: float,
        label: str,
    ) -> np.ndarray:
        """Generate the signal contribution from a single fault.

        Each defect hit is modelled as a short exponential impulse
        repeating at the defect frequency, with amplitude proportional
        to damage severity.  Up to 3 harmonics are included.
        """
        freq = order * self.shaft_freq
        component = np.zeros_like(t)

        # Impulse train via narrow raised-cosine pulses
        n_harmonics = 3
        for h in range(1, n_harmonics + 1):
            amp = severity * 0.5 / h
            # Add slight frequency jitter (slip)
            jitter = 1.0 + self._rng.normal(0, 0.002)
            component += amp * np.cos(2 * np.pi * h * freq * jitter * t)

        # Amplitude modulation at shaft frequency (for inner race / ball faults)
        if label in ("BPFI", "BSF"):
            mod = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * self.shaft_freq * t))
            component *= mod

        # High-frequency resonance burst (bearing natural frequency ~2-5 kHz)
        resonance_freq = self._rng.uniform(2000, 5000)
        resonance = severity * 0.3 * np.sin(2 * np.pi * resonance_freq * t)
        # Gate resonance to defect impulse timing
        gate = np.clip(np.cos(2 * np.pi * freq * t), 0, 1) ** 4
        component += resonance * gate

        return component

    def _advance_damage(self) -> None:
        for key in self._damage.severity:
            rate = self._damage.progression_rate[key]
            if rate > 0:
                self._damage.severity[key] = min(
                    1.0, self._damage.severity[key] + rate
                )
