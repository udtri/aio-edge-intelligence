"""MQTT bridge that subscribes to sensor topics, buffers data in sliding windows,
runs inference through the model provider, and publishes results.

Supports both plain Mosquitto brokers and Azure IoT Operations (AIO) brokers
with TLS client certificates.
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
from collections import deque
from datetime import datetime
from typing import Any

import paho.mqtt.client as mqtt

from config import AppConfig
from schemas import SensorData

logger = logging.getLogger(__name__)


class MQTTBridge:
    """Bi-directional MQTT bridge for sensor data ingestion and result publishing."""

    def __init__(self, config: AppConfig, provider: Any | None = None) -> None:
        self._config = config
        self._provider = provider

        # Sliding window buffers keyed by sensor_id.
        self._buffers: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

        # paho-mqtt v2 client
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="inference-server",
            protocol=mqtt.MQTTv5,
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the MQTT broker, optionally with TLS and SAT auth for AIO."""
        if self._config.mqtt_use_tls:
            tls_context = ssl.create_default_context()
            # AIO deployments typically inject certs via volume mounts.
            tls_context.check_hostname = False
            tls_context.verify_mode = ssl.CERT_NONE
            self._client.tls_set_context(tls_context)
            logger.info("TLS enabled for MQTT connection")

        # AIO broker SAT authentication
        if self._config.mqtt_sat_token_path:
            try:
                with open(self._config.mqtt_sat_token_path, "r") as f:
                    sat_token = f.read().strip()
                self._client.username_pw_set("K8S-SAT", sat_token)
                logger.info("SAT auth enabled from %s", self._config.mqtt_sat_token_path)
            except Exception:
                logger.exception("Failed to read SAT token from %s", self._config.mqtt_sat_token_path)

        logger.info(
            "Connecting to MQTT broker at %s:%d",
            self._config.mqtt_broker_host,
            self._config.mqtt_broker_port,
        )
        self._client.connect(
            self._config.mqtt_broker_host,
            self._config.mqtt_broker_port,
        )
        self._client.loop_start()

    def disconnect(self) -> None:
        """Gracefully disconnect from the MQTT broker."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("Disconnected from MQTT broker")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe(self) -> None:
        """Subscribe to all configured sensor topics."""
        for topic in self._config.mqtt_topics_subscribe:
            self._client.subscribe(topic)
            logger.info("Subscribed to topic: %s", topic)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: mqtt.ReasonCode,
        properties: Any = None,
    ) -> None:
        if reason_code == 0:
            logger.info("Connected to MQTT broker")
            self.subscribe()
        else:
            logger.error("MQTT connection failed: %s", reason_code)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: mqtt.ReasonCode,
        properties: Any = None,
    ) -> None:
        logger.warning("Disconnected from MQTT broker (rc=%s)", reason_code)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        """Handle an incoming MQTT message containing sensor data."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            sensor_data = SensorData(**payload)
        except Exception:
            logger.exception("Failed to parse MQTT message on topic %s", msg.topic)
            return

        sensor_id = sensor_data.sensor_id or msg.topic
        values = sensor_data.values

        with self._lock:
            if sensor_id not in self._buffers:
                self._buffers[sensor_id] = deque(maxlen=self._config.window_size)
            buf = self._buffers[sensor_id]
            buf.extend(values)

            if len(buf) >= self._config.window_size:
                window = list(buf)
                # Retain overlap samples for the next window.
                keep = self._config.window_overlap
                buf.clear()
                if keep > 0:
                    for v in window[-keep:]:
                        buf.append(v)
                self._process_window(sensor_id, window)

    # ------------------------------------------------------------------
    # Inference & publishing
    # ------------------------------------------------------------------

    def _process_window(self, sensor_id: str, window: list[float]) -> None:
        """Run inference on a full sliding window and publish the result."""
        if self._provider is None:
            logger.warning("No model provider available; skipping inference")
            return

        try:
            sensor_data = SensorData(
                values=window,
                sensor_id=sensor_id,
                timestamp=datetime.utcnow(),
            )
            result = self._provider.detect_anomalies(sensor_data)
            self.publish_result(sensor_id, result)
        except Exception:
            logger.exception("Inference failed for sensor %s", sensor_id)

    def publish_result(self, sensor_id: str, result: Any) -> None:
        """Publish an inference result to the configured MQTT results topic."""
        topic = f"{self._config.mqtt_topic_results}/{sensor_id}"
        try:
            payload = result.model_dump_json() if hasattr(result, "model_dump_json") else json.dumps(result)
            self._client.publish(topic, payload)
            logger.debug("Published result to %s", topic)
        except Exception:
            logger.exception("Failed to publish result to %s", topic)
