.PHONY: setup demo data eda smoke test train tuned experiments

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e .

demo:
	.venv/bin/python scripts/prepare_data.py --demo

data:
	.venv/bin/python scripts/prepare_data.py

eda:
	.venv/bin/python scripts/run_eda.py

smoke:
	.venv/bin/python scripts/train.py --config configs/smoke.yaml --models persistence xgboost lstm gru transformer --tag smoke

test:
	.venv/bin/python -m unittest discover -s tests -v

train:
	.venv/bin/python scripts/train.py --config configs/base.yaml

tuned:
	.venv/bin/python scripts/train.py --config configs/tuned_transformer.yaml --models transformer --tag transformer_tuned

experiments:
	.venv/bin/python scripts/run_experiments.py configs/base.yaml
