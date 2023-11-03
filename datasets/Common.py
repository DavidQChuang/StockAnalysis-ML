import pandas as pd

from dataclasses import dataclass
import inspect

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
    indicators  : list[str] = None
    
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
    function     : str = 'SMA'
    period      : int = 20
    period2     : int = 12
    
from torch.utils.data.dataset import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len=0, out_seq_len=0, conf=None):
        if conf == None:
            self.conf = DatasetConfig(seq_len, out_seq_len)
        else:
            self.conf = conf
            
        self.df: pd.DataFrame = df
        self.series_close: pd.Series = self.df['close']

    def __len__(self):
        return len(self.df) - self.conf.seq_len - self.conf.out_seq_len + 1
    
    def __getitem__(self, index):
        input = self.series_close[index: index + self.conf.seq_len]
        output = self.series_close[index + self.conf.seq_len: index + self.conf.seq_len + self.conf.out_seq_len]
        
        return { 'X': input.values, 'y': output.values }