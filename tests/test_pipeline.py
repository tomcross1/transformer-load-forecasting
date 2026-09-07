import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from load_forecasting.data import make_demo
from load_forecasting.dataset import load_splits
from load_forecasting.models import TransformerForecaster


FEATURES = [
    "global_active_power", "global_reactive_power", "voltage", "global_intensity",
    "sub_metering_1", "sub_metering_2", "sub_metering_3", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]


class PipelineTest(unittest.TestCase):
    def test_windows_scaler_and_model_shapes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.csv"
            make_demo(path, n_hours=400)
            _, splits, scaler, target_idx = load_splits(
                str(path), FEATURES, "global_active_power", 48, 12, 0.7, 0.15
            )
            x, y = splits["train"][0]
            self.assertEqual(tuple(x.shape), (48, len(FEATURES)))
            self.assertEqual(tuple(y.shape), (12,))
            self.assertTrue(np.isclose(scaler.mean[target_idx], 2.0, atol=1.0))
            model = TransformerForecaster(len(FEATURES), 12, d_model=16, nhead=2,
                                          num_layers=1, dim_feedforward=32)
            output = model(x.unsqueeze(0))
            self.assertEqual(tuple(output.shape), (1, 12))


if __name__ == "__main__":
    unittest.main()
