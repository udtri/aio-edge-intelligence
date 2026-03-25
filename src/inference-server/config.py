"""Configuration management for the inference server.

Settings are resolved in the following priority order (highest first):
1. Environment variables
2. Values from an optional ``config.yaml`` file
3. Field defaults defined below
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_CONFIG_YAML_PATH = Path("config.yaml")


def _load_yaml_config(path: Path = _CONFIG_YAML_PATH) -> dict[str, Any]:
    """Load configuration values from a YAML file if it exists."""
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            logger.info("Loaded configuration from %s", path)
            return data
        logger.warning("config.yaml did not contain a mapping; ignoring")
        return {}
    except Exception:
        logger.exception("Failed to read config.yaml; falling back to defaults")
        return {}


class AppConfig(BaseSettings):
    """Application-wide configuration backed by environment variables and YAML."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
    )

    # ── Model settings ───────────────────────────────────────────────
    model_provider: str = Field(
        default="moment",
        validation_alias="MODEL_PROVIDER",
        description="Model provider backend to use (e.g. moment)",
    )
    model_name: str = Field(
        default="AutonLab/MOMENT-1-large",
        validation_alias="MODEL_NAME",
        description="HuggingFace model identifier or local path",
    )
    model_device: str = Field(
        default="auto",
        validation_alias="MODEL_DEVICE",
        description="Device for inference: auto, cuda, or cpu",
    )

    # ── MQTT settings ────────────────────────────────────────────────
    mqtt_broker_host: str = Field(
        default="localhost",
        validation_alias="MQTT_BROKER_HOST",
    )
    mqtt_broker_port: int = Field(
        default=1883,
        validation_alias="MQTT_BROKER_PORT",
    )
    mqtt_use_tls: bool = Field(
        default=False,
        validation_alias="MQTT_USE_TLS",
    )
    mqtt_topics_subscribe: list[str] = Field(default=["sensors/#"])
    mqtt_topic_results: str = Field(default="ai/results")

    # ── Sliding-window settings ──────────────────────────────────────
    window_size: int = Field(
        default=512,
        ge=1,
        validation_alias="WINDOW_SIZE",
        description="Number of samples in the sliding window fed to the model",
    )
    window_overlap: int = Field(
        default=0,
        ge=0,
        validation_alias="WINDOW_OVERLAP",
    )

    # ── Task / server settings ───────────────────────────────────────
    default_task: str = Field(
        default="anomaly_detection",
        validation_alias="DEFAULT_TASK",
    )
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8080)

    # ── YAML overlay ─────────────────────────────────────────────────
    @model_validator(mode="before")
    @classmethod
    def _inject_yaml(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Merge values from config.yaml underneath env-var overrides."""
        yaml_values = _load_yaml_config()
        # YAML values act as defaults; explicit env vars take precedence.
        for key, value in yaml_values.items():
            values.setdefault(key, value)
        return values
