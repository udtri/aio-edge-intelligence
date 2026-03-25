#!/usr/bin/env python3
"""
MQTT Integration Test
=====================

Publishes synthetic sensor data to an MQTT broker and subscribes to
the inference results topic to verify the full MQTT bridge pipeline.

Environment variables:
    MQTT_HOST   — broker hostname  (default: localhost)
    MQTT_PORT   — broker port      (default: 1883)
    SENSOR_ID   — device identifier (default: test-motor-01)
    TIMEOUT     — seconds to wait for results (default: 30)

Usage:
    pip install paho-mqtt
    python test_mqtt_bridge.py
"""

import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SENSOR_ID = os.getenv("SENSOR_ID", "test-motor-01")
TIMEOUT = int(os.getenv("TIMEOUT", "30"))

PUB_TOPIC = f"sensors/vibration/{SENSOR_ID}"
SUB_TOPIC = "ai/results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_paho():
    try:
        import paho.mqtt.client as mqtt  # noqa: F401
        return True
    except ImportError:
        print("❌ paho-mqtt is not installed.")
        print("   Install with: pip install paho-mqtt")
        sys.exit(1)


def generate_sensor_payload(seq: int) -> dict:
    """Create a single sensor reading with a subtle anomaly at seq 15–20."""
    t = seq * 0.1
    vibration = math.sin(t) + 0.5 * math.sin(3 * t)
    if 15 <= seq <= 20:
        vibration += 3.5  # injected anomaly

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensor_id": SENSOR_ID,
        "seq": seq,
        "vibration": round(vibration, 4),
        "temperature": round(45.0 + 2.0 * math.sin(t * 0.3), 2),
        "pressure": round(101.3 + 0.5 * math.sin(t * 0.2), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    check_paho()
    import paho.mqtt.client as mqtt

    results_received: list[dict] = []
    connected_event = threading.Event()

    # -- callbacks ----------------------------------------------------------

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"✅ Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
            client.subscribe(SUB_TOPIC)
            print(f"   Subscribed to: {SUB_TOPIC}")
            connected_event.set()
        else:
            print(f"❌ Connection failed with code {rc}")
            sys.exit(1)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = msg.payload.decode()
        results_received.append(payload)
        print(f"\n📥 Result on [{msg.topic}]:")
        print(f"   {json.dumps(payload, indent=2) if isinstance(payload, dict) else payload}")

    # -- client setup -------------------------------------------------------

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"🔌 Connecting to {MQTT_HOST}:{MQTT_PORT} …")
    try:
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    except ConnectionRefusedError:
        print(f"❌ Cannot connect to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
        print("   Make sure Mosquitto or another broker is running.")
        sys.exit(1)

    client.loop_start()

    if not connected_event.wait(timeout=10):
        print("❌ Timed out waiting for MQTT connection.")
        sys.exit(1)

    # -- publish sensor data ------------------------------------------------

    num_messages = 30
    print(f"\n📤 Publishing {num_messages} sensor readings to [{PUB_TOPIC}] …")
    for seq in range(num_messages):
        payload = generate_sensor_payload(seq)
        client.publish(PUB_TOPIC, json.dumps(payload))
        if seq % 10 == 0:
            print(f"   Sent message {seq}/{num_messages}")
        time.sleep(0.1)

    print(f"   ✅ All {num_messages} messages published.")

    # -- wait for results ---------------------------------------------------

    print(f"\n⏳ Waiting up to {TIMEOUT}s for inference results on [{SUB_TOPIC}] …")
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if results_received:
            remaining = deadline - time.time()
            # Keep listening for a bit after first result
            time.sleep(min(5, max(0, remaining)))
            break
        time.sleep(1)

    # -- summary ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"   Messages published : {num_messages}")
    print(f"   Results received   : {len(results_received)}")

    if results_received:
        print("   ✅ Pipeline is working — inference results received!")
    else:
        print("   ⚠️  No results received within timeout.")
        print("   Possible causes:")
        print("   • Inference server not running")
        print("   • MQTT bridge not forwarding to inference server")
        print("   • Topics misconfigured")

    # -- cleanup ------------------------------------------------------------
    client.loop_stop()
    client.disconnect()
    print()


if __name__ == "__main__":
    main()
