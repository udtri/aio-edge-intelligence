#!/usr/bin/env python3
"""
Process Anomaly Detection Demo — Multi-Sensor Fusion
=====================================================

Generates multi-channel sensor data (temperature + pressure + vibration),
runs MOMENT anomaly detection on each channel, fuses scores, and shows how
anomalies propagate from gradual degradation to alert.

**Runs standalone** — no MQTT broker or Kubernetes required.

Prerequisites
-------------
    pip install momentfm matplotlib numpy

Usage
-----
    python process_anomaly_demo.py

What this demo does
-------------------
1. Simulates a 3-channel manufacturing process (furnace temperature,
   hydraulic pressure, vibration) using realistic sensor profiles.
2. Walks through four phases:
   - Normal operation
   - Gradual degradation (calibration drift + slow leak)
   - Anomaly peak (thermal shock + cavitation + vibration spike)
   - Recovery / alert summary
3. Runs MOMENT reconstruction-based anomaly detection per channel.
4. Fuses per-channel scores via a ``SensorFusionEngine``.
5. Visualises results with matplotlib (falls back to a formatted table
   if matplotlib is unavailable).
"""

from __future__ import annotations

import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make sensor profiles importable (they live in the simulator package)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILES_DIR = _REPO_ROOT / "src" / "sensor-simulator" / "profiles"

sys.path.insert(0, str(_PROFILES_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from temperature_furnace import TemperatureFurnaceProfile  # noqa: E402
from pressure_hydraulic import PressureHydraulicProfile    # noqa: E402
from multi_sensor_fusion import SensorFusionEngine         # noqa: E402

# ---------------------------------------------------------------------------
# Optional imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    import torch
    from momentfm import MOMENTPipeline

    _HAS_MOMENT = True
except ImportError:
    _HAS_MOMENT = False

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for CI / headless
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLING_RATE = 10.0  # Hz (shared across channels)
WINDOW_SIZE = 512     # MOMENT default input length
N_WINDOWS = 6         # number of consecutive windows to process
ANOMALY_THRESHOLD = 0.7

# Channel weights for fusion (temperature & pressure slightly more important)
CHANNEL_WEIGHTS = {
    "temperature": 1.2,
    "pressure": 1.0,
    "vibration": 0.8,
}


# ---------------------------------------------------------------------------
# Vibration generator (simple — no dedicated profile yet)
# ---------------------------------------------------------------------------

class VibrationProfile:
    """Lightweight vibration profile for demo purposes.

    Combines base motor frequency with harmonics and optional fault mode.
    """

    def __init__(
        self,
        sampling_rate: float = 10.0,
        base_rms: float = 2.5,
        noise_level: float = 0.3,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.base_rms = base_rms
        self.noise_level = noise_level
        self._rng = np.random.default_rng()
        self._t = 0.0
        self._fault_active = False

    def generate(self, n_samples: int) -> np.ndarray:
        dt = 1.0 / self.sampling_rate
        t = self._t + np.arange(n_samples) * dt
        self._t = t[-1] + dt

        # Base vibration: motor fundamental + harmonics
        signal = (
            self.base_rms * np.sin(2 * np.pi * 1.0 * t)
            + 0.4 * self.base_rms * np.sin(2 * np.pi * 3.0 * t)
            + 0.15 * self.base_rms * np.sin(2 * np.pi * 7.0 * t)
        )

        if self._fault_active:
            # Bearing fault: impulse-like bursts at sub-harmonic
            impulse_freq = 0.4  # Hz
            impulse = 3.0 * self.base_rms * np.abs(
                np.sin(2 * np.pi * impulse_freq * t)
            ) ** 8
            signal += impulse

        signal += self._rng.normal(0, self.noise_level, n_samples)
        return signal

    def inject_anomaly(self, anomaly_type: str = "bearing_fault") -> None:
        if anomaly_type == "bearing_fault":
            self._fault_active = True


# ---------------------------------------------------------------------------
# MOMENT helpers
# ---------------------------------------------------------------------------

def _load_moment_model():
    """Load MOMENT pipeline for anomaly detection."""
    print("  Loading MOMENT model (this may take a moment on first run) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "reconstruction"},
        )
        model.init()
    print("  ✅ MOMENT model ready\n")
    return model


def _anomaly_score_moment(model, window: np.ndarray) -> float:
    """Run MOMENT reconstruction on a single window and return anomaly score.

    The score is the mean absolute reconstruction error, normalised to
    roughly [0, 1] via a sigmoid-style mapping.
    """
    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    reconstruction = output.reconstruction.squeeze().numpy()
    mae = float(np.mean(np.abs(window - reconstruction)))
    # Normalise: sigmoid mapping so typical errors → ~0.3, large → ~0.9+
    score = 1.0 / (1.0 + np.exp(-2.0 * (mae - 1.5)))
    return round(score, 4)


def _anomaly_score_fallback(window: np.ndarray) -> float:
    """Heuristic anomaly score when MOMENT is not installed.

    Uses a z-score approach: if recent values deviate significantly from
    the window mean, the score rises.
    """
    mean = np.mean(window)
    std = np.std(window) + 1e-8
    tail = window[-64:]
    z = np.mean(np.abs((tail - mean) / std))
    score = 1.0 / (1.0 + np.exp(-1.5 * (z - 2.0)))
    return round(float(score), 4)


# ---------------------------------------------------------------------------
# Demo phases
# ---------------------------------------------------------------------------

def _generate_phase_data(
    temp_profile: TemperatureFurnaceProfile,
    pressure_profile: PressureHydraulicProfile,
    vibration_profile: VibrationProfile,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Generate data from all three channels."""
    return {
        "temperature": temp_profile.generate(n_samples),
        "pressure": pressure_profile.generate(n_samples),
        "vibration": vibration_profile.generate(n_samples),
    }


def _print_header(title: str) -> None:
    width = 64
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_scores(
    window_idx: int,
    channel_scores: dict[str, float],
    fused: float,
    is_anomaly: bool,
) -> None:
    flag = "🚨 ANOMALY" if is_anomaly else "✅ Normal"
    parts = "  |  ".join(
        f"{ch}: {sc:.4f}" for ch, sc in channel_scores.items()
    )
    print(f"  Window {window_idx:>2}  │  {parts}  │  fused: {fused:.4f}  │  {flag}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_results(
    all_data: dict[str, list[np.ndarray]],
    all_scores: list[dict[str, float]],
    fused_scores: list[float],
    anomaly_flags: list[bool],
) -> None:
    """Create a multi-panel matplotlib figure."""
    if not _HAS_MPL:
        return

    n_channels = len(all_data)
    fig, axes = plt.subplots(
        n_channels + 1, 1, figsize=(14, 3 * (n_channels + 1)), sharex=True
    )

    channel_names = list(all_data.keys())
    colours = {"temperature": "#e74c3c", "pressure": "#3498db", "vibration": "#2ecc71"}
    units = {"temperature": "°C", "pressure": "bar", "vibration": "g (RMS)"}

    # Plot each channel's raw data
    for idx, ch_name in enumerate(channel_names):
        ax = axes[idx]
        windows = all_data[ch_name]
        full_signal = np.concatenate(windows)
        time_axis = np.arange(len(full_signal)) / SAMPLING_RATE
        ax.plot(
            time_axis,
            full_signal,
            color=colours.get(ch_name, "#555"),
            linewidth=0.6,
            alpha=0.9,
        )
        ax.set_ylabel(f"{ch_name}\n({units.get(ch_name, '')})", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Shade anomaly windows
        for w_idx, is_anom in enumerate(anomaly_flags):
            if is_anom:
                t_start = w_idx * WINDOW_SIZE / SAMPLING_RATE
                t_end = (w_idx + 1) * WINDOW_SIZE / SAMPLING_RATE
                ax.axvspan(t_start, t_end, color="red", alpha=0.12)

    # Bottom panel: fused anomaly score
    ax_score = axes[-1]
    window_centers = [
        (i + 0.5) * WINDOW_SIZE / SAMPLING_RATE for i in range(len(fused_scores))
    ]
    bar_colors = ["#e74c3c" if f else "#2ecc71" for f in anomaly_flags]
    ax_score.bar(window_centers, fused_scores, width=WINDOW_SIZE / SAMPLING_RATE * 0.8,
                 color=bar_colors, alpha=0.8, edgecolor="white")
    ax_score.axhline(ANOMALY_THRESHOLD, color="red", linestyle="--", linewidth=1, label="Threshold")
    ax_score.set_ylabel("Fused\nAnomaly Score", fontsize=9)
    ax_score.set_xlabel("Time (s)")
    ax_score.set_ylim(0, 1.05)
    ax_score.legend(loc="upper left", fontsize=8)
    ax_score.grid(True, alpha=0.3)

    fig.suptitle(
        "Process Anomaly Detection — Multi-Sensor Fusion Demo",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "process_anomaly_demo_output.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  📊 Plot saved to {out_path}")
    plt.close(fig)


def _print_table(
    all_scores: list[dict[str, float]],
    fused_scores: list[float],
    anomaly_flags: list[bool],
) -> None:
    """Print a formatted ASCII table of results."""
    _print_header("Results Summary")
    header = (
        f"  {'Window':>6}  │  {'Temperature':>11}  │  {'Pressure':>9}  │  "
        f"{'Vibration':>9}  │  {'Fused':>6}  │  Status"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, (scores, fused, flag) in enumerate(
        zip(all_scores, fused_scores, anomaly_flags)
    ):
        status = "🚨 ANOMALY" if flag else "✅ Normal"
        print(
            f"  {i + 1:>6}  │  "
            f"{scores.get('temperature', 0):>11.4f}  │  "
            f"{scores.get('pressure', 0):>9.4f}  │  "
            f"{scores.get('vibration', 0):>9.4f}  │  "
            f"{fused:>6.4f}  │  {status}"
        )
    print()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║   Process Anomaly Detection — Multi-Sensor Fusion Demo       ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # -- Step 1: Initialise sensor profiles ---------------------------------
    _print_header("Step 1 — Initialise Sensor Profiles")
    temp_profile = TemperatureFurnaceProfile(
        sampling_rate=SAMPLING_RATE,
        ambient_temp=25.0,
        target_temp=650.0,
        noise_level=1.5,
    )
    pressure_profile = PressureHydraulicProfile(
        sampling_rate=SAMPLING_RATE,
        setpoint=150.0,
        noise_level=0.8,
    )
    vibration_profile = VibrationProfile(
        sampling_rate=SAMPLING_RATE,
        base_rms=2.5,
        noise_level=0.3,
    )
    print("  ✅ Temperature furnace profile")
    print("  ✅ Hydraulic pressure profile")
    print("  ✅ Vibration motor profile")

    # -- Step 2: Set up fusion engine ---------------------------------------
    _print_header("Step 2 — Configure Sensor Fusion Engine")
    engine = SensorFusionEngine(
        strategy="weighted_average",
        global_threshold=ANOMALY_THRESHOLD,
    )
    engine.add_channel("temperature", weight=CHANNEL_WEIGHTS["temperature"], threshold=0.65)
    engine.add_channel("pressure",    weight=CHANNEL_WEIGHTS["pressure"],    threshold=0.70)
    engine.add_channel("vibration",   weight=CHANNEL_WEIGHTS["vibration"],   threshold=0.60)
    print(f"  {engine}")
    print("  Channels: temperature (w=1.2, t=0.65), pressure (w=1.0, t=0.70), vibration (w=0.8, t=0.60)")

    # -- Step 3: Load MOMENT model ------------------------------------------
    _print_header("Step 3 — Load Anomaly Detection Model")
    model = None
    if _HAS_MOMENT:
        try:
            model = _load_moment_model()
        except Exception as exc:
            print(f"  ⚠️  Could not load MOMENT: {exc}")
            print("  Falling back to heuristic anomaly scoring.\n")
    else:
        print("  ℹ️  momentfm not installed — using heuristic anomaly scoring.")
        print("  Install with:  pip install momentfm\n")

    score_fn = (
        (lambda w: _anomaly_score_moment(model, w))
        if model is not None
        else _anomaly_score_fallback
    )

    # -- Step 4: Run through process phases ---------------------------------
    _print_header("Step 4 — Simulate Manufacturing Process")

    # We'll run N_WINDOWS windows.  Inject faults at specific windows.
    # Window layout:
    #   0-1: Normal operation
    #   2:   Begin gradual degradation (calibration drift + slow leak)
    #   3-4: Degradation intensifies + acute fault injection
    #   5:   Recovery / post-fault

    phase_labels = {
        0: "Normal operation",
        1: "Normal operation",
        2: "Gradual degradation begins (drift + leak)",
        3: "Degradation continues — injecting acute faults",
        4: "Acute anomaly (thermal shock + cavitation + bearing fault)",
        5: "Post-fault / recovery",
    }

    all_data: dict[str, list[np.ndarray]] = {
        "temperature": [],
        "pressure": [],
        "vibration": [],
    }
    all_scores: list[dict[str, float]] = []
    fused_scores: list[float] = []
    anomaly_flags: list[bool] = []

    for w_idx in range(N_WINDOWS):
        phase = phase_labels.get(w_idx, "")
        print(f"\n  ── Window {w_idx + 1}/{N_WINDOWS}: {phase} ──")

        # Inject faults at appropriate windows
        if w_idx == 2:
            temp_profile.inject_anomaly("calibration_drift")
            pressure_profile.inject_anomaly("leak")
            print("  💉 Injected: calibration drift (temperature) + slow leak (pressure)")

        if w_idx == 4:
            temp_profile.inject_anomaly("thermal_shock")
            pressure_profile.inject_anomaly("cavitation")
            vibration_profile.inject_anomaly("bearing_fault")
            print("  💉 Injected: thermal shock + cavitation + bearing fault")

        # Generate data
        data = _generate_phase_data(
            temp_profile, pressure_profile, vibration_profile, WINDOW_SIZE
        )

        # Score each channel
        channel_scores: dict[str, float] = {}
        for ch_name, ch_data in data.items():
            score = score_fn(ch_data)
            channel_scores[ch_name] = score
            engine.update(ch_name, score)
            all_data[ch_name].append(ch_data)

        fused = engine.fused_score()
        is_anom = engine.is_anomaly()

        all_scores.append(channel_scores)
        fused_scores.append(fused)
        anomaly_flags.append(is_anom)

        _print_scores(w_idx + 1, channel_scores, fused, is_anom)

        if is_anom:
            print("  ⚡ ALERT — Fused anomaly score exceeds threshold!")
            status = engine.get_status()
            triggered = [
                c["name"]
                for c in status["channels"]
                if c["is_anomaly"]
            ]
            if triggered:
                print(f"       Triggered channels: {', '.join(triggered)}")

    # -- Step 5: Summary ----------------------------------------------------
    _print_header("Step 5 — Summary")
    print(f"  Total windows processed  : {N_WINDOWS}")
    print(f"  Anomalies detected       : {sum(anomaly_flags)}")
    print(f"  Fusion strategy          : {engine.strategy}")
    print(f"  Global threshold         : {ANOMALY_THRESHOLD}")
    print()

    status = engine.get_status()
    for ch in status["channels"]:
        print(
            f"  Channel '{ch['name']}':  latest={ch['latest_score']:.4f}  "
            f"threshold={ch['threshold']}  anomaly={ch['is_anomaly']}"
        )
    print(f"\n  Fused score: {status['fused_score']:.4f}  →  "
          f"{'🚨 ANOMALY' if status['is_anomaly'] else '✅ Normal'}")

    # -- Step 6: Visualise --------------------------------------------------
    _print_header("Step 6 — Visualisation")
    _print_table(all_scores, fused_scores, anomaly_flags)

    if _HAS_MPL:
        _plot_results(all_data, all_scores, fused_scores, anomaly_flags)
    else:
        print("  ℹ️  matplotlib not installed — skipping plot.")
        print("  Install with:  pip install matplotlib\n")

    # -- Done ---------------------------------------------------------------
    print("  ✅ Demo complete!\n")


if __name__ == "__main__":
    main()
