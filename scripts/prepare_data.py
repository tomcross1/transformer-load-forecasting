#!/usr/bin/env python3
import argparse

from load_forecasting.data import download_uci, make_demo, prepare_hourly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Generate synthetic data for a quick smoke test")
    parser.add_argument("--hours", type=int, default=1200)
    args = parser.parse_args()
    if args.demo:
        make_demo("data/processed/demo_hourly.csv", n_hours=args.hours)
    else:
        raw = download_uci("data/raw")
        prepare_hourly(raw, "data/processed/household_power_hourly.csv")


if __name__ == "__main__":
    main()

