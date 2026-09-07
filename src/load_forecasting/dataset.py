from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_target(self, y: np.ndarray, target_index: int) -> np.ndarray:
        return y * self.std[target_index] + self.mean[target_index]


class WindowDataset(Dataset):
    """Samples x[t-lookback:t] -> target[t:t+horizon]."""

    def __init__(self, values: np.ndarray, prediction_starts: np.ndarray,
                 lookback: int, horizon: int, target_index: int):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.starts = prediction_starts.astype(np.int64)
        self.lookback = lookback
        self.horizon = horizon
        self.target_index = target_index

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        t = int(self.starts[i])
        x = self.values[t - self.lookback:t]
        y = self.values[t:t + self.horizon, self.target_index]
        return x, y


def load_splits(path: str, features: list[str], target: str, lookback: int,
                horizon: int, train_ratio: float, val_ratio: float):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing configured columns: {sorted(missing)}")
    values = df[features].to_numpy(dtype=np.float32)
    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= lookback or val_end - train_end <= horizon or n - val_end <= horizon:
        raise ValueError("Series is too short for the configured lookback/horizon and splits.")
    scaler = Standardizer.fit(values[:train_end])  # fit on training data only
    scaled = scaler.transform(values).astype(np.float32)
    target_index = features.index(target)
    ranges = {
        "train": np.arange(lookback, train_end - horizon + 1),
        "val": np.arange(train_end, val_end - horizon + 1),
        "test": np.arange(val_end, n - horizon + 1),
    }
    datasets = {
        name: WindowDataset(scaled, starts, lookback, horizon, target_index)
        for name, starts in ranges.items()
    }
    return df, datasets, scaler, target_index

