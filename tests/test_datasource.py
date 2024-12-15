import unittest

import numpy as np
import pandas as pd
from pandas.core.api import DataFrame as DataFrame

from stockml.datasets.datasources import Datasource, DatasourceConfig, Semantics

class TestDatasource(Datasource):
    def __init__(self, force_overwrite=False):
        self.config = DatasourceConfig(
            name = "test",
            symbol = "TEST",
            interval = "1min",
            column_flags = Semantics.TIMESTAMP | Semantics.OHLCV,
            generate_intervals = True
        )
            
        super().__init__(config=self.config)
    
    def _retrieve_dataframe(self, datasource_json: dict, config: DatasourceConfig, force_overwrite=False) -> DataFrame:
        # Create a date range
        date_rng = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')

        # Generate random data
        np.random.seed(42)
        data = np.random.randn(len(date_rng), 4)

        # Create a DataFrame
        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'], index=date_rng)

        # Ensure High is greater than Open and Close, and Low is less than Open and Close
        df['high'] = df[['open', 'close']].max(axis=1) + np.abs(df['high'])
        df['low'] = df[['open', 'close']].min(axis=1) - np.abs(df['low'])

        df['data'] = pd.Series(range(len(df))) # Set the date column as the index df.set_index('date', inplace=True)
        
        return df

class TestDatasourceMethods(unittest.TestCase):

    def test_intervals(self):
        datasource = TestDatasource()

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()