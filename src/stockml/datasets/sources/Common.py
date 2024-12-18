
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from functools import reduce
import inspect
import re
from typing import Any

import pandas as pd

from stockml.datasets.sources.columns import SourceColumn, IndicatorColumn, Semantics

from ._helpers.indicators import generate_indicators

@dataclass
class DatasourceConfig:
    @classmethod
    def from_dict(cls, env):
        # Parse column_flags as Semantics and include_columns as DatasourceColumn
        parse_column_flags = \
            lambda kv: (kv[0], Semantics.parse(kv[1]))              if kv[0] == "column_flags" else \
                       (kv[0], [ SourceColumn.from_dict(col_info) for col_info in kv[1] ])   if kv[0] == "include_columns" else \
                       (kv) 
        return cls(**{
            k: v for k, v in map(parse_column_flags, env.items())
            if k in inspect.signature(cls).parameters
        }) # type: ignore
        
    name            : str = ''
    symbol          : str = ''
    interval        : str = ''
    column_flags    : Semantics = Semantics.NONE
    
    # Below are values which can be inherited from DatasetConfig.
    # Intervals to resample to
    resample_intervals  : list[str]              = field(default_factory = lambda: [])
    # Indicators to generate data for. Used as input columns by default but may be disabled.
    indicators          : list[dict[str, Any]]   = field(default_factory = lambda: [])
    # Other columns to generate, as well as columns from the OHLCV data to include
    include_columns     : list[SourceColumn] = field(default_factory = lambda: [])
    
    @property
    def generate_intervals(self):
        has_resamplable_data = \
            ((Semantics.OHLC in self.column_flags) or (Semantics.BIDASK in self.column_flags)) and \
            (Semantics.TIMESTAMP in self.column_flags)
        
        # If resample_intervals isn't falsy and the datasource has resamplable data, generate intervals.
        return self.resample_intervals and has_resamplable_data
    
    @property
    def generate_indicators(self):
        has_resamplable_data = \
            ((Semantics.OHLC in self.column_flags) or (Semantics.BIDASK in self.column_flags)) and \
            (Semantics.TIMESTAMP in self.column_flags)
        
        # If resample_intervals isn't falsy and the datasource has resamplable data, generate indicators.
        return self.indicators and has_resamplable_data
    
    def is_multi_source(self):
        """If the Datasource has multiple types of data (OHLC, BIDASK, NEWS) at once, it will be split into multiple datasources."""
        has_ohlc = reduce(lambda a, x: a & (x in self.column_flags), (flag for flag in Semantics.OHLC), False)
        has_bidask = reduce(lambda a, x: a & (x in self.column_flags), (flag for flag in Semantics.BIDASK), False)
        has_news = Semantics.SEM_NEWS in self.column_flags
        
        return sum([has_ohlc, has_bidask, has_news]) != 1
    
    def has_weird_flags(self):
        """If the Datasource has weird flags, such as missing one part of OHLC or BIDASK or having no TIMESTAMP, gives a warning."""
        
        warnings = []
        
        has_ohlc = reduce(lambda a, x: a & (x in self.column_flags), (flag for flag in Semantics.OHLC), False)
        has_bidask = reduce(lambda a, x: a & (x in self.column_flags), (flag for flag in Semantics.BIDASK), False)
        has_news = Semantics.SEM_NEWS in self.column_flags
        
        if has_ohlc:
            warnings.append("Datasource has some OHLC flags but is missing at least one.")
        
        if has_bidask:
            warnings.append("Datasource has some BIDASK flags but is missing at least one.")
            
        if not has_ohlc and not has_bidask and not has_news:
            warnings.append("Datasource has no data columns.")
        
        if Semantics.TIMESTAMP not in self.column_flags:
            warnings.append("Datasource has no TIMESTAMP flag. Most operations require timestamps.")
            
        return warnings
    
class Source(ABC):
    def __init__(
            self,
            source_json: dict[str, dict]={},
            config=None,
            force_overwrite=False
        ):
        self.scaled_columns: set[str]
        
        self.config = config or DatasourceConfig.from_dict(source_json)
        
        if self.config is None:
            raise ValueError("Datasource failed to initialize due to missing config.")
        
        # Generate the dataframe, then verify that necessary columns are present and resample data
        self.df = self._retrieve_dataframe(source_json, self.config, force_overwrite)
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
    
    @property
    def columns(self):
        return self.df.columns
    
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
    def _retrieve_dataframe(self, source_json: dict, config: DatasourceConfig, force_overwrite=False) -> pd.DataFrame:
        """Retrieves data from the data source and parses it into a standardized dataframe.
        Results must contain columns with names corresponding to the column flags without the SEM_ prefix.
        Timestamp column must be of type datetime64[ns] (see pandas.to_datetime).
        """
        pass
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df
    
    def resample_data(self) -> pd.DataFrame:
        """Resamples the standardized dataframe into different intervals and generates indicators. 

        Returns:
            DataFrame: New DataFrame containing resampled data.
        """
        self.verify_df_interval(self.config.column_flags)
        self.verify_df_columns(self.config.column_flags)
        
        new_df = self.resample_intervals(self.config.resample_intervals)
        
        if new_df:
            return new_df
        else:
            return self.df
    
    def get_col_name(self, col, interval):
        return f"{self.config.name}_{col}_{interval}"
    
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
            # Don't generate indicators or intervals if no OHLC or bid/ask data.
            return
        
        # TODO: cut stuff into 'blocks' and do group operations to reduce lines of code
        # Prepare dataframe for resampling
        base_interval = self.config.interval
        base_td_interval = self.parse_td_interval(base_interval)
        base_columns = self.df.columns
        data_columns = self.config.column_flags.get_columns()
        
        # Get base data columns to resample.
        # This will only include columns like OHLC data or BIDASK data.
        # Columns in the base columns but not in include_columns will be dropped.
        # Other data in include_columns will be included only at the end if NEWS data.
        columns_to_resample = []
        for col in data_columns:
            if col in self.config.include_columns: #TODO: handle news
                columns_to_resample.append(col)
        
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
                    resampled_df = generated_df[columns_to_resample].resample(interval).agg(aggregator).dropna() # type: ignore
                    
                    # Generate indicators for resampling interval (and rename them)
                    resampled_df, ind_columns, ind_infos = self.generate_indicators(resampled_df)
                    
                    # Append '_interval' to column names
                    old_columns = columns_to_resample + ind_columns
                    new_columns = [ self.get_col_name(col, interval) for col in old_columns ]
                    
                    # Add scaled columns to list of scaled columns
                    scaled_columns.update(self.get_col_name(col, interval) for col in base_scaled_columns if col in columns_to_resample)
                    scaled_columns.update(self.get_col_name(col, interval) for col, info in zip(ind_columns, ind_infos) if info.is_scaled )
                    
                    # Add new columns to master df
                    generated_df[new_columns] = resampled_df[old_columns]
                    
                else:
                    # Tick resampling
                    # TODO: implement tick resampling
                    raise ValueError("TODO: Tick resampling not supported")
            del interval
            
        # Generate indicators for base interval
        generated_df, ind_columns, ind_infos = self.generate_indicators(generated_df)
        
        # Drop base columns not in the list of included columns 
        included_columns = set(col_info.name for col_info in self.config.include_columns)
        generated_df.drop([col for col in base_columns if col not in included_columns], axis=1, inplace=True)
        
        # Rename old base columns to include interval
        columns_to_rename = { col: self.get_col_name(col, base_interval) for col in (columns_to_resample + ind_columns) }
        generated_df.rename(columns_to_rename, inplace=True)
        
        # Generate datetime values
        generated_df = self.generate_datetime(generated_df)
        
        # Check that all the data columns to include actually exist
        for col in columns_to_resample:
            if col not in generated_df:
                raise ValueError("Failed resampling of Datasource. Column '{col}' is an included column but was not generated.")
        
        
        # Add scaled columns to list of scaled columns
        scaled_columns.update(self.get_col_name(col, base_interval) for col in base_scaled_columns)
        scaled_columns.update(self.get_col_name(col, base_interval) for col, info in zip(ind_columns, ind_infos) if info.is_scaled )
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