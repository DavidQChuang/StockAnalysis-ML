
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from enum import Flag, auto
import inspect
import re

import pandas as pd

from datasets.datasources.columns import DatasourceColumn, IndicatorColumn

from .Indicators import generate_indicators

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
        # Parse column_flags as Semantics and include_columns as DatasourceColumn
        parse_column_flags = \
            lambda kv: (kv[0], Semantics.parse(kv[1]))              if kv[0] == "column_flags" else \
                       (kv[0], DatasourceColumn.from_dict(kv[1]))   if kv[0] == "include_columns" else \
                       (kv) 
        return cls(**{
            k: v for k, v in map(parse_column_flags, env.items())
            if k in inspect.signature(cls).parameters
        }) # type: ignore
        
    name            : str = ''
    symbol          : str = ''
    interval        : str = ''
    column_flags    : Semantics = Semantics.NONE
    
    generate_intervals : bool = False
    
    # Below are values which can be inherited from DatasetConfig.
    # Intervals to resample to
    resample_intervals  : list[str]     = [] 
    # Indicators to generate data for. Used as input columns by default but may be disabled.
    indicators          : list[dict]    = [] 
    # Other columns to generate, as well as columns from the OHLCV data to include
    include_columns     : list[DatasourceColumn]    = [] 
    
class Datasource(ABC):
    def __init__(
            self,
            datasource_json: dict[str, dict]={},
            config=None,
            force_overwrite=False
        ):
        self.scaled_columns: set[str]
        
        self.config = config or DatasourceConfig.from_dict(datasource_json)
        
        # Generate the dataframe, then verify that necessary columns are present and resample data
        self.df = self._retrieve_dataframe(datasource_json, self.config, force_overwrite)
        self.df = self.resample_data()
    
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
    def get_subclass(cls, name):
        for subclass in cls.__subclasses__():
            if subclass.__name__ == name:
                return subclass
        return None
    
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
    def _retrieve_dataframe(self, datasource_json: dict, config: DatasourceConfig, force_overwrite=False) -> pd.DataFrame:
        """Retrieves data from the data source and parses it into a standardized dataframe.
        Results must contain columns with names corresponding to the column flags without the SEM_ prefix.
        Timestamp column must be of type datetime64[ns] (see pandas.to_datetime).
        """
        pass
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df
    
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
        
        # TODO: cut stuff into 'blocks' and do group operations to reduce lines of code
        # Prepare dataframe for resampling
        base_interval = self.config.interval
        base_td_interval = self.parse_td_interval(base_interval)
        data_columns = self.config.column_flags.get_columns()
        included_data_columns = [ col_info.name for col_info in self.config.include_columns ]
        excluded_data_columns = [ col for col in data_columns if col not in included_data_columns ]
        
        # Keep track of columns which need to be scaled
        base_scaled_columns = [ col_info.name for col_info in self.config.include_columns
                               if col_info.is_scaled and col_info.is_input ]
        scaled_columns = set[str]()
        
        # Resampling process is different for tick data and fixed-interval
        if base_td_interval is None:
            raise ValueError("Failed resampling of Datasource due to malformed base interval. Consider enabling verify_df in resample_data, which should catch this first.")
        elif type(base_td_interval) == int:
            # TODO: implement tick resampling
            raise ValueError("TODO: Tick resampling not supported")
        
        # Use timestamp index for resampling
        generated_df: pd.DataFrame = self.df.set_index('timestamp', inplace=False)
        
        # Don't resample to different intervals if disabled
        if self.config.generate_intervals:
            # Generate every resampling interval
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
                    # The usual interval format is compatible with resample().
                    resampled_df = generated_df.resample(interval).agg(aggregator).dropna() # type: ignore
                    
                    # Generate indicators for resampling interval (and rename them)
                    resampled_df, ind_columns, ind_infos = self.generate_indicators(resampled_df)
                    
                    # Append '_interval' to column names
                    new_columns =  [ f"{col}_{interval}" for col in (included_data_columns + ind_columns) ]
                    
                    # Add scaled columns to list of scaled columns
                    scaled_columns.update(f"{col}_{interval}" for col in base_scaled_columns)
                    scaled_columns.update(f"{col}_{interval}" for col, info in zip(ind_columns, ind_infos) if info.is_scaled )
                    
                    # Add new columns to master df
                    generated_df[new_columns] = resampled_df[included_data_columns + ind_columns]
                    
                else:
                    # Tick resampling
                    # TODO: implement tick resampling
                    raise ValueError("TODO: Tick resampling not supported")
            
        # Generate indicators for base interval
        generated_df, ind_columns, ind_infos = self.generate_indicators(generated_df)
        
        # Drop base columns not in the list of included columns 
        generated_df.drop(excluded_data_columns, axis=1, inplace=True)
        
        # Rename old base columns to include interval
        columns_to_rename = { col: f"{col}_{base_interval}" for col in (included_data_columns + ind_columns) }
        generated_df.rename(columns_to_rename, inplace=True)
        
        # Generate datetime values
        generated_df = self.generate_datetime(generated_df)
        
        # Check that all the data columns to include actually exist
        for col in included_data_columns:
            if col not in generated_df:
                raise ValueError("Failed resampling of Datasource. Column '{col}' is an included column but was not generated.")
        
        
        # Add scaled columns to list of scaled columns
        scaled_columns.update(f"{col}_{base_interval}" for col in base_scaled_columns)
        scaled_columns.update(f"{col}_{base_interval}" for col, info in zip(ind_columns, ind_infos) if info.is_scaled )
        self.scaled_columns = scaled_columns
        
        return generated_df
    
    def generate_indicators(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[IndicatorColumn]]:
        """Generate indicators for the dataframe if OHLC and timestamp data is present.
        Otherwise, make no changes.
        Indicators are defined in a list named 'indicators' in the dataset and datasources.
        Indicators with the 'is_input' field set to false may be used in custom functions, but are discarded afterwards. """
        # TODO: Indicators that are not used as input may be used in custom functions, but are discarded afterwards.
        if (Semantics.TIMESTAMP | Semantics.OHLC) in self.config.column_flags:
            return generate_indicators(df, self.config.indicators)
        else:
            return df, [], []
    
    def generate_datetime(self, df: pd.DataFrame):
        for col in self.config.include_columns:
            index: pd.DatetimeIndex = df.index # type: ignore
            
            match col.name:
                case 'dt_day':
                    df['dt_day'] = index.day
                case 'dt_month':
                    df['dt_month'] = index.month
                case 'dt_year':
                    df['dt_year'] = index.year
                case 'dt_hour':
                    df['dt_hour'] = index.hour
                case 'dt_minute':
                    df['dt_minute'] = index.minute
                case 'dt_weekday':
                    df['dt_weekday'] = index.weekday
                case 'dt_timestamp':
                    df['dt_timestamp'] = index.astype(int)
                    
        return df
    
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
    
    def __str__(self) -> str:
        # TODO
        
        return super().__str__()