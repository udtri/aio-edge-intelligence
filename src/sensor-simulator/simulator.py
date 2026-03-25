"""Sensor Simulator — main entry point.

Reads configuration from environment variables, instantiates sensor
profiles, and publishes their data to MQTT topics in an infinite loop
with graceful shutdown support.

Environment variables
---------------------
MQTT_BROKER_HOST : str   — Broker hostname (required).
MQTT_BROKER_PORT : int   — Broker port (default 1883).
MQTT_USE_TLS     : str   — "true" to enable TLS (default "false").
SENSOR_INTERVAL_MS : int — Publish interval in ms (default 100).
ANOMALY_PROBABILITY : float — Per-tick chance of anomaly injection (default 0.05).
SENSOR_INSTANCES : str   — Comma-separated list of profile names to run
                           (default "vibration,temperature,pressure,audio").
"""

import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from mqtt_publisher import MQTTPublisher
from profiles import get_profile

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("simulator")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MQTT_BROKER_HOST: str = os.environ.get("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT: int = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
MQTT_USE_TLS: bool = os.environ.get("MQTT_USE_TLS", "false").lower() == "true"
SENSOR_INTERVAL_MS: int = int(os.environ.get("SENSOR_INTERVAL_MS", "100"))
ANOMALY_PROBABILITY: float = float(os.environ.get("ANOMALY_PROBABILITY", "0.05"))
SENSOR_INSTANCES: list[str] = os.environ.get(
    "SENSOR_INSTANCES", "vibration,temperature,pressure,audio"
).split(",")

# Mapping from profile name → (MQTT topic prefix, unit, samples per tick)
_PROFILE_META: dict[str, dict[str, Any]] = {
    "vibration": {"unit": "g", "samples": 100},
    "temperature": {"unit": "°C", "samples": 1},
    "pressure": {"unit": "bar", "samples": 10},
    "audio": {"unit": "Pa", "samples": 1200},
}

# Anomaly types per profile
_ANOMALY_TYPES: dict[str, list[str]] = {
    "vibration": ["impact", "looseness", "gear_mesh_fault"],
    "temperature": ["runaway_heating", "cooling_failure", "thermal_shock", "calibration_drift"],
    "pressure": ["leak", "blockage", "cavitation"],
    "audio": [],  # audio uses inject_fault instead of inject_anomaly
}

_AUDIO_FAULTS = ["outer_race", "inner_race", "ball", "cage"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum: int, _frame: Any) -> None:
    global _shutdown_requested
    logger.info("Received signal %d — shutting down gracefully", signum)
    _shutdown_requested = True


def _build_message(
    sensor_id: str,
    values: np.ndarray,
    unit: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a JSON-serialisable sensor message."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensor_id": sensor_id,
        "values": values.tolist(),
        "unit": unit,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("=== Sensor Simulator Starting ===")
    logger.info(
        "Broker: %s:%d | Interval: %d ms | Anomaly prob: %.2f",
        MQTT_BROKER_HOST,
        MQTT_BROKER_PORT,
        SENSOR_INTERVAL_MS,
        ANOMALY_PROBABILITY,
    )
    logger.info("Profiles: %s", ", ".join(SENSOR_INSTANCES))

    # --- MQTT connection ----------------------------------------------------
    publisher = MQTTPublisher(client_id=f"sensor-sim-{uuid.uuid4().hex[:8]}")
    try:
        publisher.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, use_tls=MQTT_USE_TLS)
    except ConnectionError:
        logger.error("Could not connect to MQTT broker — exiting")
        sys.exit(1)

    # --- Instantiate sensors ------------------------------------------------
    rng = np.random.default_rng()
    sensors: list[dict[str, Any]] = []

    for profile_name in SENSOR_INSTANCES:
        profile_name = profile_name.strip()
        meta = _PROFILE_META.get(profile_name)
        if meta is None:
            logger.warning("Unknown profile '%s' — skipping", profile_name)
            continue
        sensor_id = f"{profile_name}-{uuid.uuid4().hex[:6]}"
        profile = get_profile(profile_name)
        sensors.append({
            "id": sensor_id,
            "name": profile_name,
            "profile": profile,
            "topic": f"sensors/{profile_name}/{sensor_id}",
            "unit": meta["unit"],
            "samples": meta["samples"],
        })
        logger.info("Created sensor: %s → %s", sensor_id, f"sensors/{profile_name}/{sensor_id}")

    if not sensors:
        logger.error("No valid sensor profiles configured — exiting")
        publisher.disconnect()
        sys.exit(1)

    # --- Publish loop -------------------------------------------------------
    interval_s = SENSOR_INTERVAL_MS / 1000.0
    tick = 0

    logger.info("Entering publish loop (Ctrl+C to stop)")
    try:
        while not _shutdown_requested:
            loop_start = time.monotonic()

            for sensor in sensors:
                profile = sensor["profile"]
                profile_name = sensor["name"]

                # Random anomaly injection
                if rng.random() < ANOMALY_PROBABILITY:
                    if profile_name == "audio":
                        fault = rng.choice(_AUDIO_FAULTS)
                        profile.inject_fault(fault)
                        logger.info("[%s] Injecting audio fault: %s", sensor["id"], fault)
                    else:
                        anomaly_choices = _ANOMALY_TYPES.get(profile_name, [])
                        if anomaly_choices:
                            anomaly = rng.choice(anomaly_choices)
                            profile.inject_anomaly(anomaly)
                            logger.info("[%s] Injecting anomaly: %s", sensor["id"], anomaly)

                # Generate and publish
                values = profile.generate(sensor["samples"])
                message = _build_message(
                    sensor_id=sensor["id"],
                    values=values,
                    unit=sensor["unit"],
                    metadata={"tick": tick, "profile": profile_name},
                )
                publisher.publish_sensor_data(sensor["topic"], message)

            tick += 1
            if tick % 100 == 0:
                logger.info("Published %d ticks across %d sensors", tick, len(sensors))

            # Sleep for the remainder of the interval
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, interval_s - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        logger.info("Shutting down — disconnecting from MQTT")
        publisher.disconnect()
        logger.info("=== Sensor Simulator Stopped ===")


if __name__ == "__main__":
    main()
