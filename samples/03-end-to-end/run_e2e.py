#!/usr/bin/env python3
"""
End-to-End Pipeline Test
========================

Checks prerequisites, runs the sensor simulator, monitors inference
results, and prints a summary report.

Works against both **standalone** (Mosquitto) and **AIO-connected**
setups.

Environment variables:
    MQTT_HOST               — broker hostname  (default: localhost)
    MQTT_PORT               — broker port      (default: 1883)
    INFERENCE_URL           — health endpoint   (default: http://localhost:5000/health)
    DURATION                — test duration in seconds (default: 60)
    ALERT_THRESHOLD         — anomaly score threshold for alerts (default: 0.8)

Usage:
    pip install paho-mqtt requests
    python run_e2e.py
"""

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:5000/health")
DURATION = int(os.getenv("DURATION", "60"))
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.8"))
SENSOR_ID = "e2e-test-motor"

PUB_TOPIC = f"sensors/vibration/{SENSOR_ID}"
SUB_TOPIC = "ai/results"

# ---------------------------------------------------------------------------
# Prerequisite Checks
# ---------------------------------------------------------------------------

def check_command(name: str) -> bool:
    return shutil.which(name) is not None


def check_mqtt_broker() -> bool:
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=5)
        client.disconnect()
        return True
    except Exception:
        return False


def check_inference_server() -> bool:
    try:
        import requests
        resp = requests.get(INFERENCE_URL, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def run_prerequisite_checks() -> bool:
    print("=" * 60)
    print("  Prerequisite Checks")
    print("=" * 60)

    all_ok = True

    # Docker
    docker_ok = check_command("docker")
    print(f"  {'✅' if docker_ok else '⚠️ '} Docker          : {'found' if docker_ok else 'not found (optional)'}")

    # Python
    print(f"  ✅ Python          : {sys.version.split()[0]}")

    # paho-mqtt
    try:
        import paho.mqtt.client  # noqa: F401
        print("  ✅ paho-mqtt       : installed")
    except ImportError:
        print("  ❌ paho-mqtt       : NOT installed  →  pip install paho-mqtt")
        all_ok = False

    # requests
    try:
        import requests  # noqa: F401
        print("  ✅ requests        : installed")
    except ImportError:
        print("  ❌ requests        : NOT installed  →  pip install requests")
        all_ok = False

    # MQTT broker connectivity
    broker_ok = check_mqtt_broker()
    print(f"  {'✅' if broker_ok else '❌'} MQTT broker      : {MQTT_HOST}:{MQTT_PORT} "
          f"{'reachable' if broker_ok else 'UNREACHABLE'}")
    if not broker_ok:
        all_ok = False

    # Inference server
    inference_ok = check_inference_server()
    print(f"  {'✅' if inference_ok else '⚠️ '} Inference server : {INFERENCE_URL} "
          f"{'healthy' if inference_ok else 'not reachable (will skip result validation)'}")

    print()
    return all_ok


# ---------------------------------------------------------------------------
# Sensor Simulator
# ---------------------------------------------------------------------------

def sensor_simulator(client, stop_event: threading.Event):
    """Publish sensor data at ~10 Hz until stop_event is set."""
    import paho.mqtt.client as mqtt  # noqa: F811

    seq = 0
    while not stop_event.is_set():
        t = seq * 0.1
        vibration = math.sin(t) + 0.5 * math.sin(3 * t) + 0.1 * math.sin(7 * t)

        # Inject anomalies periodically (every ~20s block)
        cycle_pos = seq % 200
        if 150 <= cycle_pos <= 165:
            vibration += 3.0 + 0.5 * math.sin(t * 10)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensor_id": SENSOR_ID,
            "seq": seq,
            "vibration": round(vibration, 4),
            "temperature": round(55.0 + 5.0 * math.sin(t * 0.05), 2),
            "pressure": round(101.3 + 1.0 * math.sin(t * 0.03), 2),
        }
        client.publish(PUB_TOPIC, json.dumps(payload))
        seq += 1
        time.sleep(0.1)

    return seq


# ---------------------------------------------------------------------------
# Result Collector
# ---------------------------------------------------------------------------

class ResultCollector:
    def __init__(self):
        self.results: list[dict] = []
        self.alerts: list[dict] = []
        self._lock = threading.Lock()

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        with self._lock:
            self.results.append(payload)
            score = payload.get("anomaly_score", 0)
            if isinstance(score, (int, float)) and score > ALERT_THRESHOLD:
                self.alerts.append(payload)
                print(f"  🚨 ALERT — anomaly score {score:.4f} at "
                      f"{payload.get('timestamp', 'unknown')}")

    def summary(self):
        with self._lock:
            scores = [
                r["anomaly_score"]
                for r in self.results
                if isinstance(r.get("anomaly_score"), (int, float))
            ]
            return {
                "total_inferences": len(self.results),
                "alerts_triggered": len(self.alerts),
                "avg_anomaly_score": sum(scores) / len(scores) if scores else None,
                "max_anomaly_score": max(scores) if scores else None,
                "min_anomaly_score": min(scores) if scores else None,
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        End-to-End Pipeline Test                          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # 1 — prerequisites ----------------------------------------------------
    if not run_prerequisite_checks():
        print("❌ Some prerequisites are missing. Fix the issues above and retry.")
        sys.exit(1)

    import paho.mqtt.client as mqtt

    # 2 — connect to broker ------------------------------------------------
    collector = ResultCollector()
    connected = threading.Event()

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(SUB_TOPIC)
            connected.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = collector.on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    if not connected.wait(timeout=10):
        print("❌ Failed to connect to MQTT broker.")
        sys.exit(1)

    print(f"✅ Connected to MQTT broker — publishing to [{PUB_TOPIC}]")
    print(f"   Listening for results on [{SUB_TOPIC}]")

    # 3 — run simulator ----------------------------------------------------
    print(f"\n🚀 Running sensor simulator for {DURATION}s …")
    stop = threading.Event()
    sim_thread = threading.Thread(target=sensor_simulator, args=(client, stop), daemon=True)
    sim_thread.start()

    start = time.time()
    try:
        while time.time() - start < DURATION:
            elapsed = int(time.time() - start)
            inf_count = len(collector.results)
            sys.stdout.write(
                f"\r   ⏱  {elapsed:>3}s / {DURATION}s  |  "
                f"inferences: {inf_count}  |  alerts: {len(collector.alerts)}    "
            )
            sys.stdout.flush()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    finally:
        stop.set()
        sim_thread.join(timeout=5)

    # 4 — summary ----------------------------------------------------------
    summary = collector.summary()
    print("\n")
    print("=" * 60)
    print("  End-to-End Test Summary")
    print("=" * 60)
    print(f"  Duration             : {DURATION}s")
    print(f"  Total inferences     : {summary['total_inferences']}")
    print(f"  Alerts triggered     : {summary['alerts_triggered']}")

    if summary["avg_anomaly_score"] is not None:
        print(f"  Avg anomaly score    : {summary['avg_anomaly_score']:.4f}")
        print(f"  Max anomaly score    : {summary['max_anomaly_score']:.4f}")
        print(f"  Min anomaly score    : {summary['min_anomaly_score']:.4f}")
    else:
        print("  Anomaly scores       : (none received)")

    print(f"  Alert threshold      : {ALERT_THRESHOLD}")
    print()

    if summary["total_inferences"] > 0:
        print("  ✅ Pipeline is operational — inferences were produced.")
    else:
        print("  ⚠️  No inferences received. Possible causes:")
        print("     • Inference server is not running or not processing data")
        print("     • MQTT bridge is not forwarding sensor data")
        print("     • Topic mismatch between simulator and inference server")

    # cleanup
    client.loop_stop()
    client.disconnect()
    print()


if __name__ == "__main__":
    main()
