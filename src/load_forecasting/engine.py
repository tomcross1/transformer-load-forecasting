from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader


def train_neural(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, epochs: int, learning_rate: float,
                 weight_decay: float, patience: int, checkpoint: str | Path):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item() * len(x)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_total += loss_fn(model(x), y).item() * len(x)
        train_loss = train_total / len(train_loader.dataset)
        val_loss = val_total / len(val_loader.dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch={epoch:02d} train={train_loss:.5f} val={val_loss:.5f}")
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping")
                break
    model.load_state_dict(best_state)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return history


def predict_neural(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            predictions.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    eps = 1e-6
    return {
        "MAE": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
        "RMSE": float(mean_squared_error(y_true.ravel(), y_pred.ravel()) ** 0.5),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100),
    }

