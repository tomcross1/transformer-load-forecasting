#!/usr/bin/env python3
import argparse
import pandas as pd

from load_forecasting.plots import plot_eda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/household_power_hourly.csv")
    parser.add_argument("--output", default="artifacts/eda")
    args = parser.parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"], index_col="timestamp")
    print(df.describe().round(3))
    print("\nMissing values:\n", df.isna().sum())
    plot_eda(df, args.output)
    print(f"EDA figures saved to {args.output}")


if __name__ == "__main__":
    main()

