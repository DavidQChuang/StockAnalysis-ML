import unittest

import pandas as pd
import torch.nn as nn

from stockml.datasets.Common import TimeSeriesDataset
from stockml.models.Common import PytorchModel


class TestPytorchModel(unittest.TestCase):
    def test_scale(self):
        """Tests the scaling methods to make sure they are scaling using the correct factors."""
        model = PytorchModel(nn.Linear(1, 1), {"columns": [{"name": "close"}, {"name": "volume"}]})
        dataset = TimeSeriesDataset(
            pd.DataFrame({"close": [1, 2, 3, 4], "volume": [300, 400, 500, 200]}), columns=["close", "volume"]
        )
        dataset2 = TimeSeriesDataset(
            pd.DataFrame({"close": [1, 2, 3, 4], "volume": [300, 400, 500, 200]}), columns=["close", "volume"]
        )

        # Scale dataset and fit scaler
        model.scale_dataset(dataset, True)

        # Scale original value using scale_input, which should use factors from the previous fit.
        scaled1 = model.scale_input(dataset2.df.loc[0, "close"])
        scaled2 = model.scale_input(dataset2.df.loc[0, "volume"], "volume")
        print("Scaled [0, 'close']: ", scaled1)
        print("Scaled [0, 'volume']: ", scaled2)

        # The manually scaled value should be equal to the same value in the previously scaled and fitted dataset.
        self.assertEqual(scaled1, dataset.df.loc[0, "close"])
        self.assertEqual(scaled2, dataset.df.loc[0, "volume"])

        # Unscale scaled value
        unscaled1 = model.scale_output(scaled1)
        unscaled2 = model.scale_output(scaled2, "volume")
        print("Unscaled [0, 'close']: ", unscaled1)
        print("Unscaled [0, 'volume']: ", unscaled2)

        # Should be same as the start
        self.assertEqual(unscaled1, dataset2.df.loc[0, "close"])
        self.assertEqual(unscaled2, dataset2.df.loc[0, "volume"])

    def test_data(self):
        model = PytorchModel(nn.Linear(1, 1), {"columns": [{"name": "close"}, {"name": "volume"}]})

        dataset = TimeSeriesDataset(
            pd.DataFrame({"close": [1, 2, 3, 4], "volume": [300, 400, 500, 200]}),
            seq_len=2,
            out_seq_len=2,
            columns=["close", "volume"],
        )

        model.scale_dataset(dataset, True)

        train, test = model.get_training_data(dataset)
        for data in train:
            X, Y = data["X"], data["y"]

            print(X[:, :, 0], Y)
            print(model.scale_output(X[:, :, 0]), model.scale_output(Y))


if __name__ == "__main__":
    unittest.main()
