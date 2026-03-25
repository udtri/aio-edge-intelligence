"""Download public industrial datasets for training and evaluation."""

import os
import sys
import math
import shutil
import zipfile
import urllib.request
import urllib.error
import argparse
import csv
import random

DATASETS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(DATASETS_DIR, "data")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: str, description: str = "") -> bool:
    """Download a file from *url* to *dest_path* with a text progress bar.

    Returns ``True`` on success, ``False`` on failure (prints a message).
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    label = description or os.path.basename(dest_path)

    print(f"Downloading {label} ...")
    print(f"  URL : {url}")
    print(f"  Dest: {dest_path}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            chunk_size = 1024 * 64  # 64 KB

            with open(dest_path, "wb") as fp:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fp.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(downloaded, total)

        print()  # newline after progress bar
        print(f"  ✓ Saved ({_human_size(os.path.getsize(dest_path))})")
        return True

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"\n  ✗ Download failed: {exc}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def _print_progress(downloaded: int, total: int | None) -> None:
    """Render a simple text progress bar to stdout."""
    bar_len = 40
    if total and total > 0:
        frac = downloaded / total
        filled = int(bar_len * frac)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = frac * 100
        print(
            f"\r  [{bar}] {pct:5.1f}%  {_human_size(downloaded)}/{_human_size(total)}",
            end="",
            flush=True,
        )
    else:
        print(f"\r  Downloaded {_human_size(downloaded)}", end="", flush=True)


def _human_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    for unit in ("KB", "MB", "GB"):
        nbytes /= 1024
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
    return f"{nbytes:.1f} TB"


def _extract_zip(zip_path: str, dest_dir: str) -> None:
    """Extract a ZIP archive into *dest_dir*."""
    print(f"  Extracting to {dest_dir} ...")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  ✓ Extracted {len(os.listdir(dest_dir))} items")


# ---------------------------------------------------------------------------
# NASA C-MAPSS
# ---------------------------------------------------------------------------

CMAPSS_URL = "https://data.nasa.gov/download/xaut-bemq/application%2Fzip"
CMAPSS_MIRROR = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"


def download_cmapss() -> None:
    """Download the NASA C-MAPSS Turbofan Engine Degradation dataset.

    Expected contents after extraction::

        datasets/data/cmapss/
        ├── train_FD001.txt … train_FD004.txt
        ├── test_FD001.txt  … test_FD004.txt
        └── RUL_FD001.txt   … RUL_FD004.txt

    .. note::

        The NASA data portal may require browser-based download or impose
        rate limits.  If the automated download fails, follow the manual
        instructions printed to the console.
    """
    dest_dir = os.path.join(DATA_DIR, "cmapss")
    zip_path = os.path.join(DATA_DIR, "cmapss.zip")

    if os.path.isdir(dest_dir) and any(f.startswith("train_FD") for f in os.listdir(dest_dir)):
        print("C-MAPSS dataset already present — skipping download.")
        return

    print("=" * 60)
    print("NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset")
    print("=" * 60)

    success = download_file(CMAPSS_URL, zip_path, "C-MAPSS (NASA)")

    if not success:
        print("\n  Trying mirror URL …")
        success = download_file(CMAPSS_MIRROR, zip_path, "C-MAPSS (mirror)")

    if success:
        try:
            _extract_zip(zip_path, dest_dir)
            # Some archives nest files inside a sub-folder — flatten if needed
            _flatten_single_subdir(dest_dir)
            os.remove(zip_path)
            print("  ✓ C-MAPSS dataset ready.\n")
            return
        except zipfile.BadZipFile:
            print("  ✗ Downloaded file is not a valid ZIP.")
            if os.path.exists(zip_path):
                os.remove(zip_path)

    # Manual fallback instructions
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  Automated download failed.  Please download manually:  ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  1. Open your browser and go to:                        ║")
    print("  ║     https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-  ║")
    print("  ║     Simulated-Data/ff5v-kuh6                            ║")
    print("  ║  2. Download the ZIP file.                              ║")
    print(f"  ║  3. Extract into: {dest_dir:<39s}║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()


# ---------------------------------------------------------------------------
# CWRU Bearing
# ---------------------------------------------------------------------------

# Representative subset: normal baseline + one fault per category at 0 HP load
CWRU_BASE_URL = "https://engineering.case.edu/bearingdatacenter/download-data-file"
CWRU_FILES = {
    # (filename, description, direct-download URL or Matlab key)
    "normal_0.mat": {
        "url": "https://engineering.case.edu/bearingdatacenter/48k-Drive-End-Bearing-Fault-Data",
        "desc": "Normal baseline (0 HP)",
    },
    "IR007_0.mat": {
        "url": "https://engineering.case.edu/bearingdatacenter/48k-Drive-End-Bearing-Fault-Data",
        "desc": "Inner race fault 0.007 in",
    },
    "OR007@6_0.mat": {
        "url": "https://engineering.case.edu/bearingdatacenter/48k-Drive-End-Bearing-Fault-Data",
        "desc": "Outer race fault 0.007 in @6",
    },
    "B007_0.mat": {
        "url": "https://engineering.case.edu/bearingdatacenter/48k-Drive-End-Bearing-Fault-Data",
        "desc": "Ball fault 0.007 in",
    },
}


def download_cwru_bearing() -> None:
    """Download the CWRU Bearing dataset.

    The CWRU Bearing Data Center hosts individual ``.mat`` files behind a
    web interface that does not easily support automated bulk download.
    This function attempts to fetch representative files; if that fails it
    prints manual instructions.

    Expected contents after download::

        datasets/data/cwru/
        ├── normal_0.mat
        ├── IR007_0.mat
        ├── OR007@6_0.mat
        └── B007_0.mat
    """
    dest_dir = os.path.join(DATA_DIR, "cwru")

    if os.path.isdir(dest_dir) and any(f.endswith(".mat") for f in os.listdir(dest_dir)):
        print("CWRU Bearing dataset already present — skipping download.")
        return

    print("=" * 60)
    print("CWRU Bearing Fault Dataset")
    print("=" * 60)

    os.makedirs(dest_dir, exist_ok=True)

    # Direct .mat download URLs (12-kHz drive-end data)
    direct_urls = {
        "normal_0.mat": "https://engineering.case.edu/sites/default/files/97.mat",
        "IR007_0.mat": "https://engineering.case.edu/sites/default/files/105.mat",
        "OR007@6_0.mat": "https://engineering.case.edu/sites/default/files/130.mat",
        "B007_0.mat": "https://engineering.case.edu/sites/default/files/118.mat",
    }

    any_success = False
    for fname, url in direct_urls.items():
        dest = os.path.join(dest_dir, fname)
        ok = download_file(url, dest, fname)
        any_success = any_success or ok

    if any_success:
        print("  ✓ CWRU Bearing dataset (partial or full) ready.\n")
    else:
        print()
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║  Automated download failed.  Please download manually:  ║")
        print("  ╠══════════════════════════════════════════════════════════╣")
        print("  ║  1. Visit: https://engineering.case.edu/               ║")
        print("  ║     bearingdatacenter/download-data-file                ║")
        print("  ║  2. Download .mat files for desired fault types.        ║")
        print(f"  ║  3. Place into: {dest_dir:<41s}║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        print()


# ---------------------------------------------------------------------------
# Synthetic sample dataset (always works — no network required)
# ---------------------------------------------------------------------------

def download_sample_industrial() -> None:
    """Generate a synthetic but realistic industrial sensor dataset.

    This is a reliable fallback that always works without network access.
    It creates three CSV files with 10 000 rows each, containing realistic
    patterns and injected anomalies:

    * ``vibration_motor.csv``   — motor vibration (g)
    * ``temperature_furnace.csv`` — furnace temperature (°C)
    * ``pressure_hydraulic.csv``  — hydraulic pressure (bar)
    """
    dest_dir = os.path.join(DATA_DIR, "sample")
    os.makedirs(dest_dir, exist_ok=True)

    print("=" * 60)
    print("Generating synthetic sample industrial dataset")
    print("=" * 60)

    n_rows = 10_000
    rng = random.Random(42)

    # --- vibration_motor.csv ---
    _generate_vibration(dest_dir, n_rows, rng)

    # --- temperature_furnace.csv ---
    _generate_temperature(dest_dir, n_rows, rng)

    # --- pressure_hydraulic.csv ---
    _generate_pressure(dest_dir, n_rows, rng)

    print(f"  ✓ Sample dataset ready in {dest_dir}\n")


def _generate_vibration(dest_dir: str, n: int, rng: random.Random) -> None:
    """Motor vibration signal with bearing-wear degradation and spike anomalies."""
    path = os.path.join(dest_dir, "vibration_motor.csv")
    print(f"  Creating vibration_motor.csv ({n} rows) …")

    rows: list[list[str]] = []
    base_amp = 0.5  # g RMS baseline
    for i in range(n):
        t = i * 0.01  # 100 Hz sampling → seconds
        # Slow degradation ramp in last 20 %
        degradation = 0.0
        if i > 0.8 * n:
            degradation = 0.8 * ((i - 0.8 * n) / (0.2 * n)) ** 2

        # Vibration: sinusoidal component + noise + degradation
        value = (
            base_amp * math.sin(2 * math.pi * 29.6 * t)
            + 0.3 * math.sin(2 * math.pi * 59.2 * t)
            + degradation * math.sin(2 * math.pi * 118.4 * t)
            + rng.gauss(0, 0.05 + degradation * 0.1)
        )

        # Inject spike anomalies (~0.5 % of rows)
        if rng.random() < 0.005:
            value += rng.choice([-1, 1]) * rng.uniform(2.0, 4.0)

        rows.append([f"{t:.4f}", f"{value:.6f}"])

    _write_csv(path, ["timestamp_s", "vibration_g"], rows)


def _generate_temperature(dest_dir: str, n: int, rng: random.Random) -> None:
    """Furnace temperature with cyclic heating/cooling and drift anomalies."""
    path = os.path.join(dest_dir, "temperature_furnace.csv")
    print(f"  Creating temperature_furnace.csv ({n} rows) …")

    rows: list[list[str]] = []
    setpoint = 850.0  # °C
    temp = 25.0  # start at ambient
    for i in range(n):
        t = i  # 1-second sampling

        # Heating/cooling cycles (period ~2000 s)
        cycle_phase = (i % 2000) / 2000.0
        if cycle_phase < 0.4:
            # Ramp up
            target = setpoint
        elif cycle_phase < 0.7:
            # Hold
            target = setpoint
        else:
            # Cool down
            target = 200.0

        # Simple first-order response
        tau = 150.0
        temp += (target - temp) / tau + rng.gauss(0, 0.5)

        # Inject drift anomaly in rows 6000–6500
        anomaly = 0.0
        if 6000 <= i <= 6500:
            anomaly = 30.0 * math.sin(math.pi * (i - 6000) / 500)

        rows.append([f"{t}", f"{temp + anomaly:.2f}"])

    _write_csv(path, ["timestamp_s", "temperature_C"], rows)


def _generate_pressure(dest_dir: str, n: int, rng: random.Random) -> None:
    """Hydraulic pressure with periodic pumping and leak anomalies."""
    path = os.path.join(dest_dir, "pressure_hydraulic.csv")
    print(f"  Creating pressure_hydraulic.csv ({n} rows) …")

    rows: list[list[str]] = []
    pressure = 150.0  # bar
    leak_rate = 0.0
    for i in range(n):
        t = i * 0.1  # 10 Hz

        # Pump stroke every 50 samples → pressure sawtooth
        cycle_pos = i % 50
        if cycle_pos == 0:
            pressure = 150.0 + rng.gauss(0, 1.0)
        else:
            pressure -= 0.6 + rng.gauss(0, 0.15)

        # Gradual leak starting at row 7000
        if i >= 7000:
            leak_rate = 0.02 * ((i - 7000) / (n - 7000))
            pressure -= leak_rate * cycle_pos

        # Sudden pressure drops (anomalies ~0.3 %)
        if rng.random() < 0.003:
            pressure -= rng.uniform(20, 50)

        rows.append([f"{t:.2f}", f"{max(pressure, 0):.2f}"])

    _write_csv(path, ["timestamp_s", "pressure_bar"], rows)


def _write_csv(path: str, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    size = os.path.getsize(path)
    print(f"    → {path} ({_human_size(size)})")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _flatten_single_subdir(directory: str) -> None:
    """If *directory* contains exactly one sub-directory, move its contents up."""
    entries = os.listdir(directory)
    if len(entries) == 1:
        child = os.path.join(directory, entries[0])
        if os.path.isdir(child):
            for item in os.listdir(child):
                shutil.move(os.path.join(child, item), directory)
            os.rmdir(child)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public industrial datasets for training and evaluation."
    )
    parser.add_argument(
        "--dataset",
        choices=["cmapss", "cwru", "sample", "all"],
        default="sample",
        help="Which dataset to download (default: sample).",
    )
    args = parser.parse_args()

    if args.dataset in ("cmapss", "all"):
        download_cmapss()
    if args.dataset in ("cwru", "all"):
        download_cwru_bearing()
    if args.dataset in ("sample", "all"):
        download_sample_industrial()

    print("Done.")


if __name__ == "__main__":
    main()
