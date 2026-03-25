"""Sensor profile package — factory for all sensor simulators."""

from profiles.audio_bearing import AudioBearingProfile
from profiles.pressure_hydraulic import PressureHydraulicProfile
from profiles.temperature_furnace import TemperatureFurnaceProfile
from profiles.vibration_motor import VibrationMotorProfile

__all__ = [
    "AudioBearingProfile",
    "PressureHydraulicProfile",
    "TemperatureFurnaceProfile",
    "VibrationMotorProfile",
    "get_profile",
]

_REGISTRY: dict[str, type] = {
    "vibration": VibrationMotorProfile,
    "temperature": TemperatureFurnaceProfile,
    "pressure": PressureHydraulicProfile,
    "audio": AudioBearingProfile,
}


def get_profile(name: str, **kwargs):
    """Instantiate a sensor profile by name.

    Args:
        name: One of 'vibration', 'temperature', 'pressure', 'audio'.
        **kwargs: Forwarded to the profile constructor.

    Returns:
        An instance of the requested profile class.

    Raises:
        ValueError: If *name* is not a recognised profile.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return cls(**kwargs)
