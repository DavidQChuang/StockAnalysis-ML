from ast import Index
import numpy as np
import pandas as pd

from dataclasses import dataclass
import inspect

import torch
from torch.utils.data.dataset import Dataset
from torch.utils import data

from sklearn.preprocessing import StandardScaler

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
    
    test_split        : float = 0.1
    validation_split  : float = 0.2
    batch_size          : int = 64
    
    @property
    def column_names(self):
        return [ col['name'] for col in self.columns ]
    
    @property
    def scaled_column_names(self):
        '''
        Returns all column names where is_scaled is true or not present (i.e. default if not present is true)
        '''
        return get_scaled_column_names(self.columns)
    
    @property
    def input_column_names(self):
        '''
        Returns all column names where is_input is present and true (i.e. default if not present is false)
        '''
        return get_input_column_names(self.columns)
    
##############
# Shared code between ModelConfig and DatasetConfig
def get_scaled_column_names(columns: list[dict]):
    '''
    Returns all column names where is_scaled is true or not present (i.e. default if not present is true)
    '''
    x =  [ col['name'] for col in columns if (not 'is_scaled' in col) or (col['is_scaled']) ]
    return x

def get_input_column_names(columns: list[dict]):
    '''
    Returns all column names where is_input is present and true (i.e. default if not present is false)
    '''
    return [ col['name'] for col in columns if 'is_input' in col and col['is_input'] ]
    
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
    name         : str  = ''
    function     : str  = 'SMA'
    period       : int  = 20
    period2      : int  = 12
    is_input     : bool = False
    is_scaled    : bool = True

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 seq_len=0, out_seq_len=0,
                 test_split=0.1, validation_split=0.2,
                 batch_size=64,
                 column_names: list[str]=None, scaled_column_names: list[str]=None):
        self.df: pd.DataFrame = df
        self._seq_len = seq_len
        self._out_seq_len = out_seq_len
        self._column_names = column_names
        self._scaled_column_names = scaled_column_names
        self._test_split = test_split
        self._validation_split = validation_split
        self._batch_size = batch_size
        
        self.scaler: StandardScaler | None = None
        
        if self.column_names == None or len(self.column_names) == 0:
            raise Exception("Dataset was given no column names to use as input.")
        
    @property
    def seq_len(self) -> int:
        return self._seq_len
    
    @property
    def out_seq_len(self)-> int:
        return self._out_seq_len
    
    @property
    def test_split(self)-> float:
        return self._test_split
    
    @property
    def validation_split(self)-> float:
        return self._validation_split
    
    @property
    def batch_size(self)-> int:
        return self._batch_size
    
    @property
    def column_names(self) -> list[str]:
        return self._column_names
    
    @property
    def scaled_column_names(self) -> list[str]:
        return self._scaled_column_names
    
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
    
    def __getitem__(self, index) -> dict[str, np.ndarray]:
        # if index < 0 and -index <= self.out_seq_len:
            # raise IndexError('There are not enough output data for this sequence; out_seq_len is greater than the number of remaining data points. Use .get(x) instead if you want to get only inputs.')
        return self.get(index) # type: ignore ; return type is guaranteed
        
    def get(self, index) -> dict[str, np.ndarray | None]:
        if index < 0:
            # If index is -1, need special range indexer and output is None.
            if index == -1:
                input = self.df[self.column_names][-self.seq_len:]
                output = None
            else:
                input = self.df[self.column_names][-self.seq_len + index + 1: index + 1]
                # if not enough data for out_seq_len, output is None.
                if -index <= self.out_seq_len:
                    output = None
                else:
                    output = self.df['close'][-index + 1: -index + 1 + self.out_seq_len].values
        else:
            input = self.df[self.column_names][index: index + self.seq_len]
            output = self.df['close'][index + self.seq_len: index + self.seq_len + self.out_seq_len].values
        
        return { 'X': input.values, 'y': output } # type: ignore ; output should be np.ndarray
    
    def print_validation_split(self, dataset_len, validation_split):
        train_ratio = 1-validation_split
        train_data = int(dataset_len*train_ratio)
        
        print(f"Splitting data at a {train_ratio} ratio: {train_data}/{dataset_len-train_data}")
        
    def scale_dataset(self, scaler: StandardScaler, fit=False):
        dataset = self
        
        # Store scaler in dataset once used
        if self.scaler is not None:
            raise RuntimeError("Cannot scale a dataset multiple times. This will cause unscaling to be wrong.")
        
        self.scaler = scaler
        columns_to_scale = self._scaled_column_names
        
        # For display purposes
        preview_columns = dataset.column_names
        if 'timestamp' in dataset.df.columns:
            preview_columns += ['timestamp']
        
        # Actual scaling & fitting
        if fit:
            print('> Scaling and fitting dataset.')
            print('Before: \n', dataset.df[preview_columns][:3], 'dtype=', dataset.df['close'].dtype)
        
            dataset.df[columns_to_scale] = scaler.fit_transform(dataset.df[columns_to_scale]) # type: ignore ; this is matrixlike
        else:
            print('> Scaling dataset.')
            print('Before: \n', dataset.df[preview_columns][:3], 'dtype=', dataset.df['close'].dtype)
            
            if not hasattr(scaler, 'mean_'):
                raise RuntimeError("Scaler must be fitted before being used to scale/unscale input. Run TimeSeriesDataset.scale_dataset(scaler, columns_to_scale, fit=True) first.")
            
            dataset.df[columns_to_scale] = scaler.transform(dataset.df[columns_to_scale]) # type: ignore
        
        print('After: \n', dataset.df[dataset.column_names][:3], 'dtype=', dataset.df['close'].dtype)
        print('Sanity check (should be equal to first close value): ', self.scale_output(dataset.df['close'].iloc[0]))
        print()
        
        return dataset

    def scale_input(self, input, column:str|int=0, delta=False):
        """
        Scales an unscaled input column x into the normalized distribution the given column was fitted to.\n
        If the input is the difference between two unscaled inputs, set delta to True.
            z = (x - u) / s
        """
            
        if self.scaler is None or not hasattr(self.scaler, 'mean_') or self.scaled_column_names is None:
            raise RuntimeError("Scaler must be fitted before being used to scale/unscale input. "+
                               "self.scaler or scaled_column_names is None, or the scaler has not been fitted."+
                               "Run TimeSeriesDataset.scale_dataset(scaler, columns_to_scale, fit=True) first.")
            
        # if column is str, convert to index
        if type(column) is str:
            index:int = self.scaled_column_names.index(column)
        else:
            index:int = column # type: ignore
        
        if delta == True:
            return input / self.scaler.scale_[index] # type: ignore ; if the scaler has mean_ it should have everything else too
        else:
            return (input - self.scaler.mean_[index]) / self.scaler.scale_[index] # type: ignore
    
    def scale_output(self, output, column:str|int=0, is_delta=False):
        """
        Unscales a scaled output z corresponding to the given column into unscaled units.\n
        If the input is the difference between two scaled outputs, set delta to True.
            x = z * s + u
        """
            
        if self.scaler is None or not hasattr(self.scaler, 'mean_') or self.scaled_column_names is None:
            raise RuntimeError("Scaler must be fitted before being used to scale/unscale input. "+
                               "self.scaler or scaled_column_names is None, or the scaler has not been fitted."+
                               "Run TimeSeriesDataset.scale_dataset(scaler, columns_to_scale, fit=True) first.")
        
        # if column is str, convert to index
        if type(column) is str:
            index:int = self.scaled_column_names.index(column)
        else:
            index:int = column # type: ignore
        
        if is_delta == True:
            return output * self.scaler.scale_[index] # type: ignore ; if the scaler has mean_ it should have everything else too
        else:
            return output * self.scaler.scale_[index] + self.scaler.mean_[index] # type: ignore
        
    def get_training_data(self, validation_ratio:float, batch_size:int|None=None, pin_memory=False, device='cpu'):
        dataset = self
        
        if batch_size == None:
            batch_size = self.batch_size
        
        train_ratio = 1 - validation_ratio
        batch_size = batch_size
        
        train_data, valid_data = data.random_split(dataset, [train_ratio, validation_ratio])
        
        #https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
        # If pinning, tensors on CPU remain in non-paged memory.
        # This can speed up calls to Tensor.cuda()
        #   and allows async calls with Tensor.cuda(non_blocking=True).
        if pin_memory:
            print("Pinning")
            train_dataloader = data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=False, pin_memory=True)
            valid_dataloader = data.DataLoader(valid_data, batch_size=batch_size,
                                               shuffle=False, pin_memory=True)
        # If not pinning, use the dataset collate_fn that converts data
        #   in numpy form to tensors on the correct device.
        else:
            print("Unpinned")
            train_dataloader = data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=dataset.get_collate_fn(device=device),
                                               shuffle=False)
            valid_dataloader = data.DataLoader(valid_data, batch_size=batch_size,
                                               collate_fn=dataset.get_collate_fn(device=device),
                                               shuffle=False)
        
        return train_dataloader, valid_dataloader
        
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
                case 'dt_timestamp':
                    df['dt_timestamp'] = df['timestamp'].astype(int)
        
        if self.conf.indicators != None:
            # print(df.iloc[0:5, :])
            print()
            
        self.df: pd.DataFrame = df
        super().__init__(self.df,
                         self.conf.seq_len,
                         self.conf.out_seq_len,
                         self.conf.test_split,
                         self.conf.validation_split,
                         self.conf.batch_size,
                         self.column_names,
                         self.scaled_column_names)

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
    