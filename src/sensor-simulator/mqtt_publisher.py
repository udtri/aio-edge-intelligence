"""MQTT publishing helper for sensor simulator.

Wraps paho-mqtt with reconnection logic and JSON serialization.
"""

import json
import logging
import ssl
import time
from typing import Any

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTPublisher:
    """Publishes sensor data to an MQTT broker with automatic reconnection."""

    def __init__(self, client_id: str = "") -> None:
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish
        self._connected = False
        self._host: str = ""
        self._port: int = 1883
        self._max_retries = 10
        self._retry_delay_s = 5.0

    # -- callbacks -----------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        if reason_code == 0:
            logger.info("Connected to MQTT broker %s:%d", self._host, self._port)
            self._connected = True
        else:
            logger.error("MQTT connection refused: %s", reason_code)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any = None,
        reason_code: Any = None,
        properties: Any = None,
    ) -> None:
        logger.warning("Disconnected from MQTT broker (rc=%s)", reason_code)
        self._connected = False

    @staticmethod
    def _on_publish(
        client: mqtt.Client,
        userdata: Any,
        mid: int,
        reason_code: Any = None,
        properties: Any = None,
    ) -> None:
        logger.debug("Message %d published", mid)

    # -- public API ----------------------------------------------------------

    def connect(
        self,
        host: str,
        port: int = 1883,
        use_tls: bool = False,
        keepalive: int = 60,
    ) -> None:
        """Connect to the MQTT broker with retry logic.

        Args:
            host: Broker hostname or IP.
            port: Broker port (default 1883, or 8883 for TLS).
            use_tls: Enable TLS with default system CA bundle.
            keepalive: MQTT keepalive interval in seconds.
        """
        self._host = host
        self._port = port

        if use_tls:
            self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            logger.info("TLS enabled for MQTT connection")

        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    "Connecting to %s:%d (attempt %d/%d)",
                    host,
                    port,
                    attempt,
                    self._max_retries,
                )
                self._client.connect(host, port, keepalive=keepalive)
                self._client.loop_start()
                # Wait briefly for the on_connect callback
                deadline = time.monotonic() + 5.0
                while not self._connected and time.monotonic() < deadline:
                    time.sleep(0.1)
                if self._connected:
                    return
                logger.warning("Connection attempt %d timed out", attempt)
            except OSError as exc:
                logger.warning("Connection attempt %d failed: %s", attempt, exc)
            time.sleep(self._retry_delay_s)

        raise ConnectionError(
            f"Failed to connect to MQTT broker at {host}:{port} "
            f"after {self._max_retries} attempts"
        )

    def publish_sensor_data(
        self,
        topic: str,
        data: dict[str, Any],
        qos: int = 1,
    ) -> None:
        """Serialize *data* to JSON and publish to *topic*.

        Args:
            topic: MQTT topic string.
            data: Dictionary payload (must be JSON-serialisable).
            qos: MQTT QoS level (0, 1, or 2).
        """
        if not self._connected:
            logger.warning("Not connected — attempting reconnect before publish")
            try:
                self._client.reconnect()
                time.sleep(1.0)
            except OSError as exc:
                logger.error("Reconnect failed: %s", exc)
                return

        payload = json.dumps(data, default=str)
        info = self._client.publish(topic, payload, qos=qos)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(
                "Publish to %s failed (rc=%d)", topic, info.rc
            )

    def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""
        logger.info("Disconnecting from MQTT broker")
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
