import unittest
from datasets.Common import TimeSeriesDataset
from models.Common import PytorchModel

from models.SimpleLSTM import SimpleLSTM
import torch.nn as nn
import pandas as pd

class TestStringMethods(unittest.TestCase):

    def test_scale(self):
        '''
        Tests the scaling methods to make sure they are scaling using the correct factors.
        '''
        
        model = PytorchModel(nn.Linear(1,1), { 'columns': [
            { "name": "close" },
            { "name": "volume" }
        ]})
        dataset = TimeSeriesDataset(pd.DataFrame({'close': [1,2,3,4], 'volume': [300,400,500,200]}), column_names=['close', 'volume'])
        dataset2 = TimeSeriesDataset(pd.DataFrame({'close': [1,2,3,4], 'volume': [300,400,500,200]}), column_names=['close', 'volume'])
        
        # Scale dataset and fit scaler
        model.scale_dataset(dataset, True)
        
        # Scale original value using scale_input, which should use factors from the previous fit.
        scaled1 = model.scale_input(dataset2.df.loc[0, "close"])
        scaled2 = model.scale_input(dataset2.df.loc[0, "volume"], 'volume')
        print("Scaled [0, 'close']: ", scaled1)
        print("Scaled [0, 'volume']: ", scaled2)
        
        # The manually scaled value should be equal to the same value in the previously scaled and fitted dataset.
        self.assertEqual(scaled1, dataset.df.loc[0, "close"])
        self.assertEqual(scaled2, dataset.df.loc[0, "volume"])
        
        # Unscale scaled value
        unscaled1 = model.scale_output(scaled1)
        unscaled2 = model.scale_output(scaled2, 'volume')
        print("Unscaled [0, 'close']: ", unscaled1)
        print("Unscaled [0, 'volume']: ", unscaled2)
        
        # Should be same as the start
        self.assertEqual(unscaled1, dataset2.df.loc[0, "close"])
        self.assertEqual(unscaled2, dataset2.df.loc[0, "volume"])

if __name__ == '__main__':
    unittest.main()