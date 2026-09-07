#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from load_forecasting.dataset import load_splits
from load_forecasting.engine import metrics, predict_neural, train_neural
from load_forecasting.models import build_model
from load_forecasting.plots import plot_forecast, plot_history, plot_metric_comparison
from load_forecasting.utils import get_device, load_config, save_json, seed_everything


def arrays(dataset):
    xs, ys = zip(*(dataset[i] for i in range(len(dataset))))
    return np.stack([x.numpy() for x in xs]), np.stack([y.numpy() for y in ys])


def compact_xgb_features(x: np.ndarray, target_index: int) -> np.ndarray:
    """Recent target lags + per-feature moments; avoids a huge flattened design matrix."""
    recent = x[:, -24:, target_index]
    means, stds = x.mean(axis=1), x.std(axis=1)
    last = x[:, -1, :]
    return np.concatenate([recent, means, stds, last], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--models", nargs="+", default=["persistence", "xgboost", "lstm", "gru", "transformer"])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--no-positional-encoding", action="store_true")
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.lookback is not None:
        cfg["data"]["lookback"] = args.lookback
    if args.d_model is not None:
        cfg["model"]["d_model"] = args.d_model
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
    if args.target_only:
        cfg["data"]["features"] = [cfg["data"]["target"]]
    seed_everything(cfg["seed"])
    dc, tc = cfg["data"], cfg["training"]
    _, datasets, scaler, target_idx = load_splits(
        dc["processed_path"], dc["features"], dc["target"], dc["lookback"], dc["horizon"],
        dc["train_ratio"], dc["val_ratio"],
    )
    output = Path("artifacts") / args.tag
    output.mkdir(parents=True, exist_ok=True)
    print("samples:", {k: len(v) for k, v in datasets.items()})
    device = get_device()
    print("device:", device)
    results = {}
    horizon_rows = []
    test_x, test_y = arrays(datasets["test"])
    y_true = scaler.inverse_target(test_y, target_idx)
    for name in args.models:
        started = time.time()
        if name == "persistence":
            pred_scaled = np.repeat(test_x[:, -1, target_idx, None], dc["horizon"], axis=1)
        elif name == "xgboost":
            train_x, train_y = arrays(datasets["train"])
            model = MultiOutputRegressor(XGBRegressor(
                objective="reg:squarederror", random_state=cfg["seed"], n_jobs=1, **cfg["xgboost"]
            ), n_jobs=1)
            model.fit(compact_xgb_features(train_x, target_idx), train_y)
            pred_scaled = model.predict(compact_xgb_features(test_x, target_idx))
        else:
            loaders = {
                split: DataLoader(ds, batch_size=tc["batch_size"], shuffle=(split == "train"),
                                  num_workers=tc["num_workers"])
                for split, ds in datasets.items()
            }
            model = build_model(name, len(dc["features"]), dc["horizon"], cfg["model"],
                                positional_encoding=not args.no_positional_encoding)
            history = train_neural(
                model, loaders["train"], loaders["val"], device, tc["epochs"],
                tc["learning_rate"], tc["weight_decay"], tc["patience"], output / f"{name}.pt",
            )
            plot_history(history, output / f"{name}_training.png")
            pred_scaled, _ = predict_neural(model, loaders["test"], device)
        y_pred = scaler.inverse_target(pred_scaled, target_idx)
        results[name] = metrics(y_true, y_pred) | {"seconds": round(time.time() - started, 2)}
        for h in range(dc["horizon"]):
            horizon_rows.append({"model": name, "horizon": h + 1, **metrics(y_true[:, h], y_pred[:, h])})
        plot_forecast(y_true, y_pred, output / f"{name}_forecast.png")
        print(name, results[name])
    table = pd.DataFrame(results).T.sort_values("RMSE")
    table.to_csv(output / "metrics.csv", index_label="model")
    pd.DataFrame(horizon_rows).to_csv(output / "metrics_by_horizon.csv", index=False)
    save_json(results, output / "metrics.json")
    plot_metric_comparison(results, output / "model_comparison.png")
    print("\n", table.round(4))


if __name__ == "__main__":
    main()
