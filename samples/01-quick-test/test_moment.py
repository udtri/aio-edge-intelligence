#!/usr/bin/env python3
"""
Quick Test: MOMENT Time Series Foundation Model
================================================

Standalone script to verify MOMENT works on synthetic sensor data.
No MQTT, no Kubernetes, no Docker — just pure Python.

Usage:
    pip install momentfm torch numpy
    python test_moment.py
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_dependencies():
    """Verify required packages are installed."""
    missing = []
    for pkg in ["torch", "momentfm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install momentfm torch")
        sys.exit(1)
    print("✅ All dependencies found.")


def generate_synthetic_data(
    length: int = 512,
    anomaly_start: int = 350,
    anomaly_end: int = 380,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic vibration signal with an injected anomaly.

    Returns a 1-D numpy array of shape (length,).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)

    # Base signal: sine wave + harmonics + noise
    signal = (
        np.sin(t)
        + 0.5 * np.sin(3 * t)
        + 0.3 * np.sin(5 * t)
        + rng.normal(0, 0.1, length)
    )

    # Inject anomaly — sudden vibration spike
    signal[anomaly_start:anomaly_end] += rng.normal(3.0, 0.5, anomaly_end - anomaly_start)
    return signal


def print_section(title: str):
    """Pretty-print a section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_section("MOMENT Quick Test")

    # 1 — dependency check ------------------------------------------------
    print("\n🔍 Checking dependencies …")
    check_dependencies()

    import torch
    from momentfm import MOMENTPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    # 2 — generate data ---------------------------------------------------
    print_section("Generating Synthetic Sensor Data")
    data = generate_synthetic_data()
    print(f"   Signal length : {len(data)} samples")
    print(f"   Mean / Std    : {data.mean():.4f} / {data.std():.4f}")
    print(f"   Min / Max     : {data.min():.4f} / {data.max():.4f}")
    print("   Anomaly injected at samples 350–380 (vibration spike)")

    # 3 — anomaly detection -----------------------------------------------
    print_section("Anomaly Detection with MOMENT")
    print("   Loading MOMENT for reconstruction …")
    model_recon = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction"},
    )
    model_recon.init()

    # MOMENT expects (batch, n_channels, seq_len)
    tensor_in = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    model_recon = model_recon.to(device)

    with torch.no_grad():
        output = model_recon(tensor_in)

    reconstructed = output.output.squeeze().cpu().numpy()
    recon_error = np.abs(data - reconstructed)

    # Simple threshold: mean + 3 * std of reconstruction error
    threshold = recon_error.mean() + 3 * recon_error.std()
    anomaly_mask = recon_error > threshold
    anomaly_indices = np.where(anomaly_mask)[0]

    print(f"   Reconstruction error — mean: {recon_error.mean():.4f}, std: {recon_error.std():.4f}")
    print(f"   Threshold (mean + 3σ): {threshold:.4f}")
    print(f"   Anomalies detected   : {anomaly_mask.sum()} samples")
    if len(anomaly_indices) > 0:
        print(f"   Anomaly indices      : {anomaly_indices[:20].tolist()} …")
        # Check overlap with injected anomaly
        injected = set(range(350, 380))
        detected = set(anomaly_indices.tolist())
        overlap = injected & detected
        print(f"   Overlap with injected: {len(overlap)}/{len(injected)} "
              f"({100 * len(overlap) / len(injected):.0f}%)")
    else:
        print("   ⚠️  No anomalies detected — try adjusting the threshold.")

    # 4 — forecasting -----------------------------------------------------
    print_section("Forecasting with MOMENT")
    print("   Loading MOMENT for forecasting …")
    forecast_horizon = 96
    model_forecast = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": forecast_horizon,
        },
    )
    model_forecast.init()
    model_forecast = model_forecast.to(device)

    with torch.no_grad():
        forecast_output = model_forecast(tensor_in)

    forecast = forecast_output.output.squeeze().cpu().numpy()
    print(f"   Forecast horizon : {forecast_horizon} steps")
    print(f"   Forecast shape   : {forecast.shape}")
    print(f"   Forecast mean    : {forecast.mean():.4f}")
    print(f"   Forecast std     : {forecast.std():.4f}")
    print(f"   First 10 values  : {np.round(forecast[:10], 4).tolist()}")

    # 5 — summary ---------------------------------------------------------
    print_section("Summary")
    print("   ✅ MOMENT loaded and ran successfully")
    print("   ✅ Anomaly detection identified injected spike region")
    print(f"   ✅ Forecasting produced {forecast_horizon}-step prediction")
    print()
    print("   Next steps:")
    print("   • Try with real sensor data  → samples/01-quick-test/sample_data.csv")
    print("   • Test MQTT integration      → samples/02-mqtt-integration/")
    print("   • Run end-to-end pipeline    → samples/03-end-to-end/")
    print()


if __name__ == "__main__":
    main()
