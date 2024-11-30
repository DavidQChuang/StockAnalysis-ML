
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from enum import Flag, auto
import inspect
from operator import index
import re

import pandas as pd

from datasets import indicators

class Semantics(Flag):
    NONE        = 0
    
    # When used as a data source descriptor, TimeSeriesDataset will expect
    # 'timestamp' column to be present if the corresponding bit is set.
    TIMESTAMP    = auto()
    
    # When used as a data source descriptor, TimeSeriesDataset will expect
    # 'bid' 'ask' columns to be present if the corresponding bit is set.
    BID         = auto()
    ASK         = auto()
    BIDASK = BID | ASK
    
    # When used as a data source descriptor, TimeSeriesDataset will expect
    # 'open' 'high' 'low' 'close' columns to be present if the corresponding bit is set.
    OPEN        = auto()
    HIGH        = auto()
    LOW         = auto()
    CLOSE       = auto()
    OHLC = OPEN | HIGH | LOW | CLOSE
    
    # When used as a data source descriptor, TimeSeriesDataset will expect
    # 'volume' column to be present if the corresponding bit is set.
    VOLUME      = auto()
    OHLCV = OHLC | VOLUME
    
    # Has no effect as a data source descriptor.
    # When data is passed to the model, these flags will be injected to the corresponding datapoints
    # as categorical data.
    SEM_PRICE       = auto()
    SEM_VOLATILITY  = auto()
    SEM_INDICATOR   = auto()
    SEM_NEWS        = auto()
    SEM_TRADE_INFO  = auto()
    SEMANTICS = SEM_PRICE | SEM_VOLATILITY | SEM_INDICATOR | SEM_NEWS | SEM_TRADE_INFO
    
    # @classmethod
    # def column_names(cls):
    #     return [ "OPEN", "HIGH", "LOW", "CLOSE", "BID", "ASK", "VOLUME" ]
    
    @classmethod
    def parse(cls, enum_str: str):
        """Parses a string in the format 'FLAG1 [& FLAG2] [& FLAG3] ...'"""
        parts = enum_str.split("&")
        enum = Semantics.NONE
        for part_str in parts:
            try:
                part = Semantics[part_str]
                enum |= part
            except KeyError:
                raise ValueError(f"Invalid flag: {part_str}")
        return enum
    
    def get_columns(self):
        # Columns which require a corresponding column in the dataframe
        required_columns = Semantics.OHLCV | Semantics.BIDASK | Semantics.TIMESTAMP
        # Columns expected to be present in the dataframe which require a corresponding column
        expected_columns = self & required_columns
        
        return [ col.name.lower() for col, _ in expected_columns ] # type: ignore
    
@dataclass
class DatasourceConfig:
    @classmethod
    def from_dict(cls, env):
        parse_column_flags = \
            lambda kv: (kv[0], Semantics.parse(kv[1])) if kv[0] == "column_flags" else kv
        
        return cls(**{
            k: v for k, v in map(parse_column_flags, env.items())
            if k in inspect.signature(cls).parameters
        }) # type: ignore
        
    name            : str = ''
    symbol          : str = ''
    interval        : str = ''
    column_flags    : Semantics = Semantics.NONE
    
    resample_intervals : list[str] = []
    generate_intervals : bool = False
    
    indicators      : list[dict] = []

class Datasource(ABC):
    def __init__(
            self,
            datasource_json: dict[str, dict],
            force_overwrite=False
        ):
        self.config = DatasourceConfig.from_dict(datasource_json)
        
        # Generate the dataframe
        self.df = self.get_dataframe(datasource_json, self.config, force_overwrite)
        
        # Verify that necessary columns are present and resample data
        self.resample_data()
    
    unit_map = {
        't': 'ticks',
        'tick': 'ticks',
        
        'sec': 'seconds',
        'second': 'seconds',
        
        'min': 'minutes',
        'minute': 'minutes',
        
        'hr': 'hours',
        'hour': 'hours',
        
        'd': 'days',
        'day': 'days',
        
        'w': 'weeks',
        'week': 'weeks'
    }
    
    @classmethod
    def parse_td_interval(cls, interval_str) -> int | timedelta | None:
        match = re.match(r'([0-9]+)(t|tick|ticks|sec|second|seconds|min|minute|minutes|hr|hour|hours|d|day|days|w|week|weeks)', interval_str)
        if match is None:
            return None
            
        value = int(match.group(1))
        unit  = match.group(2)
        unit  = cls.unit_map[unit] if unit in cls.unit_map else unit
            
        # If ticks, just return the number of ticks
        if unit == 'ticks':
            return value
        # If time unit, return the corresponding timedelta
        else:
            return timedelta(**{ unit: value })
    
    @abstractmethod
    def get_dataframe(self, datasource_json: dict, config: DatasourceConfig, force_overwrite=False) -> pd.DataFrame:
        """Retrieves data from the data source and parses it into a standardized dataframe.
        Results must contain columns with names corresponding to the column flags without the SEM_ prefix.
        Timestamp column must be of type datetime64[ns] (see pandas.to_datetime).
        """
        pass
    
    def resample_data(self):
        """Resamples the standardized dataframe into different intervals and generates indicators. 

        Returns:
            DataFrame: New DataFrame containing resampled data.
        """
        self.verify_df_interval(self.config.column_flags)
        self.verify_df_columns(self.config.column_flags)
        
        new_df = self.resample_intervals(self.config.resample_intervals)
        return new_df
    
    def resample_intervals(self, intervals):
        # Get aggregator functions
        if Semantics.OHLCV in self.config.column_flags:
            aggregator = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        elif Semantics.OHLC in self.config.column_flags:
            aggregator = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            
        # TODO: add bid/ask
        else:
            return
        
        # Prepare dataframe for resampling
        base_interval = self.config.interval
        base_td_interval = self.parse_td_interval(base_interval)
        data_columns = self.config.column_flags.get_columns()
        
        # Resampling process is different for tick data and fixed-interval
        if base_td_interval is None:
            raise ValueError("Failed resampling of Datasource due to malformed base interval. Consider enabling verify_df in resample_data, which should catch this first.")
        elif type(base_td_interval) == int:
            # TODO: implement tick resampling
            raise ValueError("TODO: Tick resampling not supported")
        
        # Map old column names to new names including interval (used later)
        # keys: original columns; values: new columns
        new_data_columns = { col: f"{col}_{base_interval}" for col in data_columns }
        
        # Use timestamp index for resampling
        new_df = self.df.set_index('timestamp')
        
        for interval in intervals:
            td_interval = self.parse_td_interval(interval)
            
            if td_interval is None:
                raise ValueError("Failed resampling of Datasource due to malformed resampling interval.")
            elif type(td_interval) == timedelta:
                # Timedelta resampling
                if td_interval <= base_td_interval: # type: ignore - for some reason fails to recognize
                    print(f"Warning: Encountered resampling interval {interval} less than or equal to base interval {base_interval}. Skipping.")
                    continue
                
                # Resample and copy new columns from resampled df to new df
                # - timestamp index will automatically match rows
                new_columns = [ f"{col}_{interval}" for col in data_columns ]
                
                # The usual interval format is compatible with resample().
                resampled_df = self.df.resample(interval).agg(aggregator).dropna() # type: ignore
                new_df[new_columns] = resampled_df[data_columns]
            else:
                # Tick resampling
                # TODO: implement tick resampling
                raise ValueError("TODO: Tick resampling not supported")
            
        # Rename old base columns to include interval
        new_df.rename(new_data_columns)
        
        return new_df
    
    def verify_df_interval(self, interval):
        # Check that timestamp column is the correct dtype and has the expected interval
        ts_col = self.df['timestamp']
        
        if ts_col.dtype.name != 'datetime64[ns]':
            raise ValueError("Failed verification of Datasource due to invalid timestamp dtype.")
        
        # Interval only needs to be checked if it is a timedelta.
        # If it's ticks, no checking is required.
        if interval is None:
            raise ValueError("Failed verification of Datasource due to malformed interval.")
        elif type(interval) == timedelta:
            deltas = ts_col.diff()
            deltas = deltas[deltas < timedelta(hours=4)]
            
            if not all(deltas == interval):
                raise ValueError(
                    "Failed verification of Datasource due to some rows failing to adhere to the time interval.")
    
    def verify_df_columns(self, format_flags: Semantics):
        """Makes sure all the flagged columns are present in the given dataframe.
        Args:
            df (DataFrame): The dataframe to check.
            format_flags (DatasetColumns): The flags for this dataframe.
                OHLCV, BID/ASK, and TIMESTAMP flags require a corresponding column name in the dataframe (in lower case).
        Returns:
            bool
        """
        
        if self.df is None:
            raise ValueError("Failed verification of Datasource due to null DataFrame.")
        
        if not self.check_df_columns(self.df, format_flags):
            raise ValueError(f"Failed verification of Datasource due to missing necessary columns for column flags {self.config.column_flags}. Present columns: {self.df.columns}")
    
    def check_df_columns(self, df: pd.DataFrame, format_flags: Semantics):
        # Gets columns expected to be in the dataframe according to the format flags
        expected_columns = format_flags.get_columns()
        
        for col in expected_columns:
            if col not in df: # type: ignore
                return False
            
        return True