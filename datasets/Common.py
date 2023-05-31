from pandas import DataFrame

from dataclasses import dataclass
import inspect

@dataclass
class StandardConfig:
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
    
from torch.utils.data.dataset import Dataset

class DataframeDataset(Dataset):
    def __init__(self, dataset_json, forceOverwrite=False):
        self.config = StandardConfig.from_dict(dataset_json)
        self.df = self._get_dataframe(dataset_json, forceOverwrite)
        self.df = self.df['close']

    def __len__(self):
        return len(self.df) - self.config.seq_len - self.config.out_seq_len + 1
    
    def __getitem__(self, index):
        input = self.df[index: index + self.config.seq_len]
        output = self.df[index + self.config.seq_len: index + self.config.seq_len + self.config.out_seq_len]
        
        return { 'X': input.values, 'y': output.values }
    
    def _get_dataframe(dataset_json, forceOverwrite=False) -> DataFrame:
        pass