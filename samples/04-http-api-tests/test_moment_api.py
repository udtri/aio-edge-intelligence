#!/usr/bin/env python3
"""End-to-end API test suite for the MOMENT inference server.

Usage:
    # With default endpoint (localhost:8080)
    python test_moment_api.py

    # With custom endpoint
    python test_moment_api.py --base-url http://localhost:9090

    # Via port-forward to AIO cluster
    kubectl port-forward svc/aio-si-aio-sensor-intelligence-inference-server 8080:8080 -n azure-iot-operations &
    python test_moment_api.py
"""

import argparse
import json
import math
import random
import sys
import time
import urllib.request
from typing import Any


class MomentAPITester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.results: dict[str, str] = {}

    def _post(self, path: str, data: dict, timeout: int = 60) -> tuple[dict, float]:
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        return body, time.time() - t0

    def _get(self, path: str, timeout: int = 10) -> tuple[dict, float]:
        req = urllib.request.Request(f"{self.base_url}{path}")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        return body, time.time() - t0

    def _record(self, name: str, passed: bool, message: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        self.results[name] = status
        icon = "✅" if passed else "❌"
        print(f"  {icon} {status}: {message}")

    def test_health(self) -> None:
        print("\n" + "=" * 60)
        print("TEST: Health Endpoint")
        print("=" * 60)
        try:
            r, lat = self._get("/health")
            print(f"  status: {r['status']}, model: {r['model']}, ready: {r['ready']}")
            print(f"  latency: {lat:.3f}s")
            self._record("health", r["ready"] is True, "Model loaded and ready")
        except Exception as e:
            print(f"  Error: {e}")
            self._record("health", False, str(e))

    def test_models(self) -> None:
        print("\n" + "=" * 60)
        print("TEST: Models Endpoint")
        print("=" * 60)
        try:
            r, lat = self._get("/models")
            print(f"  provider: {r.get('provider')}")
            print(f"  model: {r.get('model_name')}")
            print(f"  tasks: {r.get('supported_tasks')}")
            print(f"  device: {r.get('device')}")
            print(f"  status: {r.get('status')}")
            self._record("models", r.get("status") == "ready", "Model info retrieved")
        except Exception as e:
            print(f"  Error: {e}")
            self._record("models", False, str(e))

    def test_anomaly_normal(self) -> None:
        print("\n" + "=" * 60)
        print("TEST: Anomaly Detection — Normal Sine Wave (512 samples)")
        print("=" * 60)
        signal = [math.sin(i * 0.05) for i in range(512)]
        try:
            r, lat = self._post(
                "/infer/anomaly",
                {"values": signal, "channels": 1, "sensor_id": "test-normal"},
            )
            scores = r["anomaly_scores"]
            avg = sum(scores) / len(scores)
            print(f"  is_anomaly: {r['is_anomaly']}")
            print(f"  severity: {r['severity']}")
            print(f"  threshold: {r['threshold']:.6f}")
            print(f"  scores: len={len(scores)}, avg={avg:.6f}, max={max(scores):.6f}")
            print(f"  latency: {lat:.3f}s")
            self._record(
                "anomaly_normal",
                len(scores) == 512,
                f"Got {len(scores)} scores, avg={avg:.4f}",
            )
        except Exception as e:
            print(f"  Error: {e}")
            self._record("anomaly_normal", False, str(e))

    def test_anomaly_spike(self) -> None:
        print("\n" + "=" * 60)
        print("TEST: Anomaly Detection — Injected Spike (positions 350-380)")
        print("=" * 60)
        random.seed(42)
        signal = [math.sin(i * 0.05) for i in range(512)]
        for i in range(350, 380):
            signal[i] += random.gauss(3.0, 0.5)
        try:
            r, lat = self._post(
                "/infer/anomaly",
                {"values": signal, "channels": 1, "sensor_id": "test-spike"},
            )
            scores = r["anomaly_scores"]
            spike_avg = sum(scores[350:380]) / 30
            other_avg = (sum(scores[:350]) + sum(scores[380:])) / 482
            ratio = spike_avg / max(other_avg, 1e-9)
            print(f"  is_anomaly: {r['is_anomaly']}")
            print(f"  spike region avg: {spike_avg:.6f}")
            print(f"  normal region avg: {other_avg:.6f}")
            print(f"  spike/normal ratio: {ratio:.1f}x")
            print(f"  latency: {lat:.3f}s")
            self._record(
                "anomaly_spike",
                ratio > 2.0,
                f"Spike region {ratio:.1f}x higher than normal",
            )
        except Exception as e:
            print(f"  Error: {e}")
            self._record("anomaly_spike", False, str(e))

    def test_forecast(self, horizon: int = 96) -> None:
        print("\n" + "=" * 60)
        print(f"TEST: Forecasting — {horizon}-Step Horizon")
        print("=" * 60)
        signal = [math.sin(i * 0.05) for i in range(512)]
        try:
            r, lat = self._post(
                "/infer/forecast",
                {
                    "data": {
                        "values": signal,
                        "channels": 1,
                        "sensor_id": "test-forecast",
                    },
                    "forecast_horizon": horizon,
                },
            )
            fv = r.get("forecast_values", [])
            print(f"  forecast_horizon: {r.get('forecast_horizon')}")
            print(f"  values returned: {len(fv)}")
            if fv:
                print(f"  first 5: {[round(v, 4) for v in fv[:5]]}")
                print(f"  last 5:  {[round(v, 4) for v in fv[-5:]]}")
            print(f"  latency: {lat:.3f}s")
            self._record(
                "forecast",
                len(fv) == horizon,
                f"Got {len(fv)} values (expected {horizon})",
            )
        except Exception as e:
            print(f"  Error: {e}")
            self._record("forecast", False, str(e))

    def test_determinism(self) -> None:
        print("\n" + "=" * 60)
        print("TEST: Determinism — Same Input Twice")
        print("=" * 60)
        signal = [math.sin(i * 0.05) for i in range(512)]
        payload = {"values": signal, "channels": 1, "sensor_id": "det-test"}
        try:
            r1, _ = self._post("/infer/anomaly", payload)
            r2, _ = self._post("/infer/anomaly", payload)
            max_diff = max(
                abs(a - b)
                for a, b in zip(r1["anomaly_scores"], r2["anomaly_scores"])
            )
            print(f"  max score difference: {max_diff:.15f}")
            print(f"  threshold difference: {abs(r1['threshold'] - r2['threshold']):.15f}")
            self._record(
                "determinism",
                max_diff < 1e-6,
                f"Max diff={max_diff} ({'deterministic' if max_diff == 0 else 'within tolerance'})",
            )
        except Exception as e:
            print(f"  Error: {e}")
            self._record("determinism", False, str(e))

    def test_short_input(self) -> None:
        print("\n" + "=" * 60)
        print("TEST: Edge Case — Short Input (256 samples)")
        print("=" * 60)
        signal = [math.sin(i * 0.05) for i in range(256)]
        try:
            r, lat = self._post(
                "/infer/anomaly",
                {"values": signal, "channels": 1, "sensor_id": "test-short"},
            )
            n = len(r["anomaly_scores"])
            print(f"  scores returned: {n}")
            print(f"  latency: {lat:.3f}s")
            self._record("short_input", n > 0, f"Got {n} scores for 256-sample input")
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"  HTTP {e.code}: {body[:200]}")
            self._record("short_input", False, f"HTTP {e.code}")
        except Exception as e:
            print(f"  Error: {e}")
            self._record("short_input", False, str(e))

    def test_long_input(self) -> None:
        print("\n" + "=" * 60)
        print("TEST: Edge Case — Long Input (1024 samples)")
        print("=" * 60)
        signal = [math.sin(i * 0.05) for i in range(1024)]
        try:
            r, lat = self._post(
                "/infer/anomaly",
                {"values": signal, "channels": 1, "sensor_id": "test-long"},
            )
            n = len(r["anomaly_scores"])
            print(f"  scores returned: {n}")
            print(f"  latency: {lat:.3f}s")
            self._record("long_input", n > 0, f"Got {n} scores for 1024-sample input")
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"  HTTP {e.code}: {body[:200]}")
            self._record("long_input", False, f"HTTP {e.code}")
        except Exception as e:
            print(f"  Error: {e}")
            self._record("long_input", False, str(e))

    def run_all(self) -> bool:
        print(f"Running MOMENT API tests against {self.base_url}")
        print("=" * 60)

        self.test_health()
        self.test_models()
        self.test_anomaly_normal()
        self.test_anomaly_spike()
        self.test_forecast()
        self.test_determinism()
        self.test_short_input()
        self.test_long_input()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        passed = sum(1 for v in self.results.values() if v == "PASS")
        total = len(self.results)
        for name, status in self.results.items():
            icon = "✅" if status == "PASS" else "❌"
            print(f"  {icon} {name}: {status}")
        print(f"\n  Total: {passed}/{total} passed")

        return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MOMENT inference server API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of the inference server (default: http://localhost:8080)",
    )
    args = parser.parse_args()

    tester = MomentAPITester(base_url=args.base_url)
    success = tester.run_all()
    sys.exit(0 if success else 1)
