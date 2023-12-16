import numpy as np
import pandas as pd

from dataclasses import dataclass
import inspect

import torch
from torch.utils.data.dataset import Dataset

import datasets.indicators as indicators

@dataclass
class DatasetConfig:
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    # Model I/O window sizes
    # These values will be copied from the model JSON,
    # so the values of these don't matter and they don't need to be defined in 'dataset'.
    seq_len     : int = 24
    out_seq_len : int = 1
    indicators  : list[dict] = None
    columns     : list[dict] = None
    
    @property
    def column_names(self):
        return [ col['name'] for col in self.columns ]
    
    @property
    def scaled_column_names(self):
        '''
        Returns all column names where is_scaled is not false or not present.
        '''
        return [ col['name'] for col in self.columns if (not 'is_scaled' in col) or (not col['is_scaled']) ]
    
    @property
    def input_column_names(self):
        '''
        Returns all column names where is_input is present and true.
        '''
        return [ col['name'] for col in self.columns if 'is_scaled' in col and col['is_scaled'] ]
    
@dataclass
class IndicatorConfig:
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    # Model I/O window sizes
    # These values will be copied from the model JSON,
    # so the values of these don't matter and they don't need to be defined in 'dataset'.
    name         : str  = None
    function     : str  = 'SMA'
    period       : int  = 20
    period2      : int  = 12
    is_input     : bool = False
    is_scaled    : bool = True

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=0, out_seq_len=0, column_names: list[str]=None):
        self.df: pd.DataFrame = df
        self._seq_len = seq_len
        self._out_seq_len = out_seq_len
        self._column_names = column_names
        
        if self.column_names == None or len(self.column_names) == 0:
            raise Exception("Dataset was given no column names to use as input.")
        
    @property
    def seq_len(self) -> int:
        return self._seq_len
    
    @property
    def out_seq_len(self)-> int:
        return self._out_seq_len
    
    @property
    def column_names(self) -> list[str]:
        return self._column_names
    
    def get_collate_fn(self, device=None, **tensor_args):
        if device == None or device.startswith('cpu'):
            return lambda batch:{
                'X': torch.Tensor(np.array([item['X'] for item in batch ]), **tensor_args).float(),
                'y': torch.Tensor(np.array([item['y'] for item in batch ]), **tensor_args).float() }
        else:
            return lambda batch: {
                'X': torch.Tensor(np.array([item['X'] for item in batch ]), **tensor_args).to(device),
                'y': torch.Tensor(np.array([item['y'] for item in batch ]), **tensor_args).to(device) }
        
    def __len__(self):
        return len(self.df) - self.seq_len - self.out_seq_len + 1
    
    def __getitem__(self, index):
        input = self.df[self.column_names][index: index + self.seq_len]
        output = self.df['close'][index + self.seq_len: index + self.seq_len + self.out_seq_len]
        
        return { 'X': input.values, 'y': output.values }

class AdvancedTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, df: pd.DataFrame, conf=None):
        if conf == None:
            self.conf = DatasetConfig()
        else:
            self.conf = conf
        
        # Calculate indicators
        start_index = 0
        
        if self.conf.indicators != None:
            print("Loading indicators " + ','.join(map(lambda x: x['function'], self.conf.indicators)))
            for indicator in self.conf.indicators:
                ind_conf = IndicatorConfig.from_dict(indicator)
                
                # How many values to remove from the start of the array due to insufficient data points
                values_to_remove = ind_conf.period
                
                close_values = df['close']
                
                match ind_conf.function.upper():
                    case 'SMA':
                        ind_values = indicators.get_series_sma(ind_conf.period, close_values)
                        
                    case 'EMA':
                        ind_values = indicators.get_series_ema(ind_conf.period, close_values)
                        
                    case 'MACD':
                        ind_values = indicators.get_series_macd(ind_conf.period, ind_conf.period2, close_values)
                            
                    case 'LOG_VOL':
                        ind_values = indicators.get_series_log_vol(df['volume'])
                        values_to_remove = 0
                        
                    case 'DELTA_CLOSE':
                        ind_values = df['close'].diff()
                        values_to_remove = 1
                            
                    case _:
                        raise Exception("Invalid indicator name: " + ind_conf.function)
                
                # Set amount of values to remove to the largest removal size
                start_index = max(start_index, values_to_remove)

                if ind_conf.name == None:
                    ind_conf.name = indicators.get_indicator_name(ind_conf.function, ind_conf.period)

                df[ind_conf.name] = ind_values
        
        # Remove values without indicators
        if start_index != 0:
            df = df.iloc[start_index:, :]
        
        df = df.assign(timestamp = pd.to_datetime(df['timestamp']))
        for col in self.conf.columns:
            match col['name']:
                case 'dt_day':
                    df['dt_day'] = df['timestamp'].dt.day
                case 'dt_month':
                    df['dt_month'] = df['timestamp'].dt.month
                case 'dt_year':
                    df['dt_year'] = df['timestamp'].dt.year
                case 'dt_hour':
                    df['dt_hour'] = df['timestamp'].dt.hour
                case 'dt_minute':
                    df['dt_minute'] = df['timestamp'].dt.minute
        
        if self.conf.indicators != None:
            # print(df.iloc[0:5, :])
            print()
            
        self.df: pd.DataFrame = df
        super().__init__(self.df, self.conf.seq_len, self.conf.out_seq_len, self.column_names)

    @property
    def indicators(self):
        return self.conf.indicators
    
    @property
    def columns(self):
        return self.conf.columns
    
    @property
    def column_names(self):
        return self.conf.column_names
    
    @property
    def scaled_column_names(self):
        return self.conf.scaled_column_names
        
    @property
    def input_column_names(self):
        return self.conf.input_column_names
    