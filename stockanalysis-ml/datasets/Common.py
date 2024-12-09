from abc import ABC, abstractmethod
from ast import Index
from datetime import timedelta
from enum import Enum, Flag, IntEnum, auto
from typing import Any

import numpy as np
import pandas as pd

from dataclasses import dataclass
import inspect

import torch
from torch.utils.data.dataset import Dataset
from torch.utils import data

from sklearn.preprocessing import StandardScaler

from datasets.datasources import Datasource, Semantics

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
    seq_len     : int = 0
    out_seq_len : int = 0
    
    datasources         : dict[str, dict[str, Any]] = {}
    # Fields passed to datasources
    resample_intervals  : list[str]  = []
    indicators          : list[dict[str, Any]] = []
    columns             : list[dict[str, Any]] = []
    
    # Other parmaeters for use during inference training
    test_split        : float = 0.1
    validation_split  : float = 0.2
    batch_size          : int = 64
    
    target      : str = "close"
    
    @classmethod
    def get_datasource_inherited_keys(cls):
        return ['resample_intervals', 'indicators', 'columns']
    
    @classmethod
    def get_datasource_inherited_values(cls, dataset_json):
        return { key: dataset_json[key] for key in cls.get_datasource_inherited_keys() }
    
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
    Returns all column names where is_scaled is present and true (i.e. default if not present is false)
    '''
    x =  [ col['name'] for col in columns if ('is_scaled' in col) and col['is_scaled'] ]
    return x

def get_input_column_names(columns: list[dict]):
    '''
    Returns all column names where is_input is not present or true (i.e. default if not present is true)
    '''
    return [ col['name'] for col in columns if (not 'is_input' in col) or col['is_input'] ]
    
@dataclass
class DataframeConfig:
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    symbol      : str = ''
    interval    : timedelta = timedelta()
    

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 target="close",
                 seq_len=0, out_seq_len=0,
                 test_split=0.1, validation_split=0.2,
                 batch_size=64,
                 column_names: list[str]=None, scaled_column_names: list[str]=None): # type: ignore
        self.df: pd.DataFrame = df
        self._target = target
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
    def target(self)-> str:
        return self._target
    
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
                    output = self.df[self.target][-index + 1: -index + 1 + self.out_seq_len].values
        else:
            input = self.df[self.column_names][index: index + self.seq_len]
            output = self.df[self.target][index + self.seq_len: index + self.seq_len + self.out_seq_len].values
        
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
        print('Sanity check (should be equal to first close value): ',
              (dataset.df['close'].iloc[0] * self.scaler.scale_[0] + self.scaler.mean_[0])) # type: ignore
        print()
        
        return dataset
        
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
    def __init__(self, dataset_json: dict[str, Any]=None, data_sources: list[Datasource]=None, conf:DatasetConfig=None): # type: ignore
        if dataset_json is not None:
            self.conf = DatasetConfig.from_dict(dataset_json)
            
            data_sources = []
            for source_name, source_json in self.conf.datasources.items():
                if 'datasource_class' not in source_json:
                    raise Exception(f"'datasource_class' cannot be None in datasource {source_name}.")
                
                source_class_name = source_json['datasource_class']
                SourceClass = Datasource.get_subclass(source_class_name)
                
                # Use the dataset's values for these keys
                # unless they're already given inside the datasource
                copy_values = DatasetConfig.get_datasource_inherited_values(dataset_json)
                source_json = copy_values.update({source_json}) # type: ignore
                
                if SourceClass is not None:
                    source = SourceClass(source_json)
                    data_sources.append(source)
                else:
                    raise Exception(f"Class of 'datasource_class' ({source_class_name}) could not be found in datasource {source_name}.")
        else:
            self.conf = conf
        
        # Combine dataframes from data sources
        dfs = []
        all_columns = set[str]()
        for source in data_sources:
            # Get dataframe
            df = source.get_dataframe()
            columns = set(df.columns)
            
            # Make sure no columns overlap
            # TODO: necessary?
            overlapping_cols = all_columns.intersection(columns)
            if len(overlapping_cols) != 0:
                raise ValueError(f"Datasources could not be combined due to overlapping columns: {overlapping_cols}")
            
            # Update set of existing columns
            all_columns.update(columns)
            
            # Add dataframe to list of dfs
            dfs.append(df)
        
        self.df: pd.DataFrame = pd.concat(dfs, axis=1)
        
        super().__init__(self.df,
                         self.conf.target,
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