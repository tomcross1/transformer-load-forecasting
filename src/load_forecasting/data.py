from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip"
RAW_FILENAME = "household_power_consumption.txt"
MEASUREMENT_COLUMNS = [
    "global_active_power", "global_reactive_power", "voltage",
    "global_intensity", "sub_metering_1", "sub_metering_2", "sub_metering_3",
]


def download_uci(raw_dir: str | Path) -> Path:
    """Download and extract the official UCI archive (about 20 MB compressed)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / RAW_FILENAME
    if output.exists():
        print(f"Already present: {output}")
        return output
    archive = raw_dir / "household_power_consumption.zip"
    print(f"Downloading {UCI_URL}")
    urllib.request.urlretrieve(UCI_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extract(RAW_FILENAME, raw_dir)
    archive.unlink()
    print(f"Extracted: {output}")
    return output


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df = df.copy()
    for name, values, period in [
        ("hour", idx.hour, 24), ("dow", idx.dayofweek, 7), ("month", idx.month - 1, 12)
    ]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    return df


def prepare_hourly(raw_path: str | Path, output_path: str | Path) -> Path:
    """Parse UCI minute data, aggregate hourly, and fill only short gaps."""
    raw = pd.read_csv(raw_path, sep=";", na_values="?", low_memory=False)
    raw.columns = [c.lower() for c in raw.columns]
    timestamps = pd.to_datetime(
        raw.pop("date") + " " + raw.pop("time"), format="%d/%m/%Y %H:%M:%S"
    )
    raw.index = timestamps
    raw = raw.apply(pd.to_numeric, errors="coerce")
    hourly = raw.resample("h").mean()
    # Causal filling uses past observations only; long gaps are dropped.
    hourly = hourly.ffill(limit=6).dropna()
    hourly = add_calendar_features(hourly)
    hourly.index.name = "timestamp"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path)
    print(f"Saved {len(hourly):,} hourly rows to {output_path}")
    return output_path


def make_demo(output_path: str | Path, n_hours: int = 1200, seed: int = 42) -> Path:
    """Create a correlated synthetic series for fast pipeline verification only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_hours, freq="h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    daily = 0.8 * np.sin(2 * np.pi * (hour - 7) / 24)
    weekly = 0.25 * (dow < 5)
    trend = np.linspace(0, 0.25, n_hours)
    active = 2.0 + daily + weekly + trend + rng.normal(0, 0.12, n_hours)
    df = pd.DataFrame(index=idx)
    df["global_active_power"] = np.maximum(active, 0.1)
    df["global_reactive_power"] = 0.18 * active + rng.normal(0, 0.03, n_hours)
    df["voltage"] = 240 - 0.7 * active + rng.normal(0, 0.5, n_hours)
    df["global_intensity"] = 4.2 * active + rng.normal(0, 0.2, n_hours)
    for i in range(1, 4):
        df[f"sub_metering_{i}"] = np.maximum(active * (0.8 + i * 0.35) + rng.normal(0, 0.2, n_hours), 0)
    df = add_calendar_features(df)
    df.index.name = "timestamp"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved demo data to {output_path}")
    return output_path
