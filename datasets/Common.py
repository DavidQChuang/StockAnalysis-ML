import pandas as pd

from dataclasses import dataclass
import inspect

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
    is_scaled    : bool = False
    
from torch.utils.data.dataset import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=0, out_seq_len=0, column_names=None):
        self.df: pd.DataFrame = df
        self._seq_len = seq_len
        self._out_seq_len = out_seq_len
        self._column_names = column_names
        
        if self.column_names == None or len(self.column_names) == 0:
            raise Exception("Dataset was given no column names to use as input.")
        
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def out_seq_len(self):
        return self._out_seq_len
    
    @property
    def column_names(self):
        return self._column_names
        
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
                ind_values = [0] * ind_conf.period
                
                # How many values to remove from the start of the arraydue to insufficient data points
                start_index = max(start_index, ind_conf.period)
                close_values = df['close']
                
                match ind_conf.function:
                    case 'SMA':
                        ind_values = indicators.get_series_sma(ind_conf.period, close_values)
                        
                    case 'EMA':
                        ind_values = indicators.get_series_ema(ind_conf.period, close_values)
                        
                    case 'MACD':
                        ind_values = indicators.get_series_macd(ind_conf.period, ind_conf.period2, close_values)
                            
                    case _:
                        raise Exception("Invalid indicator name: " + ind_conf.function)

                if ind_conf.name == None:
                    ind_conf.name = indicators.get_indicator_name(ind_conf.function, ind_conf.period)

                df[ind_conf.name] = ind_values
        
        # Remove values without indicators
        if start_index != 0:
            df = df.iloc[start_index:, :]
        
        if self.conf.indicators != None:
            print(df.iloc[0:5, :])
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
        return [ col['name'] for col in self.conf.columns ]