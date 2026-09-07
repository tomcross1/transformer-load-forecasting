from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_eda(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    df["global_active_power"].resample("D").mean().plot(ax=axes[0], color="#2563eb")
    axes[0].set(title="Daily Mean Active Power", ylabel="kW", xlabel="")
    profile = df.groupby(df.index.hour)["global_active_power"].mean()
    profile.plot(ax=axes[1], marker="o", color="#ea580c")
    axes[1].set(title="Mean Intraday Load Profile", xlabel="Hour", ylabel="kW")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_load_patterns.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature Correlations")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_correlations.png", dpi=180)
    plt.close(fig)


def plot_forecast(y_true, y_pred, output_path: str | Path, sample: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_true[sample], label="Actual", marker="o")
    ax.plot(y_pred[sample], label="Forecast", marker="o")
    ax.set(xlabel="Forecast horizon (hours)", ylabel="Active power (kW)", title="24-step Forecast Example")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_history(history: dict, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE (standardized)", title="Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(results: dict, output_path: str | Path) -> None:
    frame = pd.DataFrame(results).T[["MAE", "RMSE"]]
    ax = frame.plot(kind="bar", figsize=(9, 4), color=["#2563eb", "#ea580c"])
    ax.set(title="Held-out Test Performance", ylabel="Error (kW)", xlabel="")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
