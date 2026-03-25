# Public Industrial Datasets

This directory contains links and instructions for public datasets you
can use to test and evaluate **aio-sensor-intelligence**.

> **Note:** These datasets are not included in the repository due to
> size.  Follow the download instructions below or use the automated
> download script.

## Quick Start — Automated Download

Use `datasets/scripts/download_datasets.py` to download datasets with a
single command (requires only the Python standard library):

```bash
# Download a synthetic sample dataset (always works, no network issues)
python datasets/scripts/download_datasets.py --dataset sample

# Download the NASA C-MAPSS turbofan dataset
python datasets/scripts/download_datasets.py --dataset cmapss

# Download the CWRU bearing fault dataset
python datasets/scripts/download_datasets.py --dataset cwru

# Download everything
python datasets/scripts/download_datasets.py --dataset all
```

The script saves files under `datasets/data/<name>/`.  If an automated
download fails (some public dataset servers restrict automated access) the
script prints clear manual-download instructions.

The **sample** dataset is a synthetic but realistic fallback that always
works offline.  It generates three CSV files (10 000 rows each) with
injected anomalies — useful for quick testing:

| File | Signal | Anomaly type |
|------|--------|-------------|
| `vibration_motor.csv` | Motor vibration (g) | Bearing-wear degradation + spike anomalies |
| `temperature_furnace.csv` | Furnace temperature (°C) | Thermal drift anomaly |
| `pressure_hydraulic.csv` | Hydraulic pressure (bar) | Gradual leak + sudden drops |

---

## NASA Turbofan Engine Degradation (C-MAPSS)

| | |
|---|---|
| **Description** | Run-to-failure simulations of turbofan engines under different operating conditions and fault modes. |
| **Sensors** | 21 sensor channels (temperatures, pressures, speeds, fuel flow, etc.) |
| **Size** | ~26 MB (4 sub-datasets: FD001–FD004) |
| **Use case** | Predictive maintenance, remaining useful life (RUL) estimation |

### Download

1. Go to the [NASA Prognostics Data Repository](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6).
2. Download the ZIP file.
3. Extract into `datasets/cmapss/`.

### Usage with This Project

```python
import pandas as pd
import numpy as np

# Load FD001 training data
cols = ['unit', 'cycle'] + [f'os{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
df = pd.read_csv('datasets/cmapss/train_FD001.txt', sep=r'\s+', header=None, names=cols)

# Extract sensor 7 (vibration-like) for unit 1
unit1 = df[df['unit'] == 1]['s7'].values

# Feed into MOMENT (requires 512-sample window)
window = unit1[:512]
```

---

## CWRU Bearing Dataset

| | |
|---|---|
| **Description** | Vibration data from bearings with seeded faults (inner race, outer race, ball defects) at various loads. |
| **Sensors** | Drive-end accelerometer (12 kHz / 48 kHz sampling) |
| **Size** | ~600 MB |
| **Use case** | Bearing fault detection, anomaly detection |

### Download

1. Visit the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file).
2. Download the `.mat` files for the desired fault types and loads.
3. Place into `datasets/cwru/`.

### Usage with This Project

```python
import scipy.io
import numpy as np

# Load a .mat file
data = scipy.io.loadmat('datasets/cwru/105.mat')

# Find the drive-end accelerometer key (varies per file)
de_key = [k for k in data.keys() if 'DE' in k and 'time' in k][0]
signal = data[de_key].flatten()

# Downsample from 12 kHz to ~10 Hz for MOMENT (or use raw for high-freq analysis)
# For quick testing, just take a 512-sample window
window = signal[:512]
```

---

## SECOM Semiconductor Manufacturing

| | |
|---|---|
| **Description** | Sensor data from a semiconductor manufacturing process with pass/fail labels. |
| **Sensors** | 590 process features per observation |
| **Size** | ~3 MB |
| **Use case** | Process anomaly detection, quality prediction |

### Download

1. Visit the [UCI Machine Learning Repository — SECOM](https://archive.ics.uci.edu/dataset/179/secom).
2. Download `secom.data` and `secom_labels.data`.
3. Place into `datasets/secom/`.

### Usage with This Project

```python
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('datasets/secom/secom.data', sep=r'\s+', header=None)
labels = pd.read_csv('datasets/secom/secom_labels.data', sep=r'\s+', header=None)

# Each row is one observation with 590 features
# For time-series analysis, treat feature index as the "time" axis
observation = data.iloc[0].dropna().values

# Or, stack multiple observations to form a time series per feature
feature_0_series = data[0].dropna().values[:512]
```

---

## Tennessee Eastman Process (TEP)

| | |
|---|---|
| **Description** | Simulation of a chemical plant with 21 programmed faults. A classic benchmark for process monitoring. |
| **Sensors** | 52 process variables (11 manipulated, 41 measured) |
| **Size** | ~150 MB |
| **Use case** | Multi-variable anomaly detection, fault classification |

### Download

1. Visit the [TEP dataset on Kaggle](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset) or the original [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1).
2. Download the simulation output files.
3. Place into `datasets/tep/`.

### Usage with This Project

```python
import pandas as pd
import numpy as np

# Load fault-free training data
df = pd.read_csv('datasets/tep/d00.dat', sep=r'\s+', header=None)

# 52 columns = 52 process variables, 500 rows = 500 time steps
# Extract variable 1 as a time series
var1 = df[0].values[:512]

# For multi-variable analysis, use all columns
all_vars = df.values[:512, :]  # shape: (512, 52)
```

---

## Directory Structure (After Download)

```
datasets/
├── README.md              ← this file
├── scripts/
│   └── download_datasets.py
├── data/
│   ├── cmapss/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   ├── RUL_FD001.txt
│   │   └── …
│   ├── cwru/
│   │   ├── normal_0.mat
│   │   ├── IR007_0.mat
│   │   └── …
│   ├── sample/
│   │   ├── vibration_motor.csv
│   │   ├── temperature_furnace.csv
│   │   └── pressure_hydraulic.csv
│   ├── secom/
│   │   ├── secom.data
│   │   └── secom_labels.data
│   └── tep/
│       ├── d00.dat
│       ├── d01.dat
│       └── …
└── custom/
    └── (your own data)
```

---

## Adding Your Own Data

To use your own sensor data:

1. **CSV format** with columns: `timestamp`, `value` (or named sensor
   columns).
2. Place in `datasets/custom/`.
3. Load and reshape to match MOMENT's expected input: a 1-D array of at
   least 512 samples.

See `samples/01-quick-test/sample_data.csv` for an example format.
