#!/usr/bin/env python3
"""Run named Transformer ablations as separate, reproducible commands."""
import subprocess
import sys


EXPERIMENTS = [
    ("transformer_full", []),
    ("transformer_no_positional", ["--no-positional-encoding"]),
    ("transformer_univariate", ["--target-only"]),
    ("transformer_72h_context", ["--lookback", "72"]),
    ("transformer_dmodel_32", ["--d-model", "32"]),
    ("transformer_dropout_0", ["--dropout", "0.0"]),
    ("transformer_tuned", ["--d-model", "32", "--dropout", "0.0"]),
]


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"
    for tag, extras in EXPERIMENTS:
        command = [sys.executable, "scripts/train.py", "--config", config,
                   "--models", "transformer", "--tag", tag, *extras]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
