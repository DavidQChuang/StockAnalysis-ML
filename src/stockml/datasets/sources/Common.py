
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from functools import reduce
import inspect
import re
from typing import Any, TypeAlias

import pandas as pd

from stockml.datasets.sources.columns import SourceColumn, IndicatorColumn, Semantics, generate_indicators

Interval: TypeAlias = int | timedelta

@dataclass
class DatasourceConfig:
    @classmethod
    def parse_columns(cls):
        return lambda kv: \
            (kv[0], Semantics.parse(kv[1])) if kv[0] == "column_flags" else \
            (kv[0], Source.parse_interval(kv[1])) if kv[0] == "interval" else \
            (kv[0], [ SourceColumn.from_dict(col_info) for col_info in kv[1] ]) if kv[0] == "include_columns" else \
            (kv[0], [ IndicatorColumn.from_dict(col_info) for col_info in kv[1] ]) if kv[0] == "indicators" else \
            (kv) 
            
    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in map(cls.parse_columns(), env.items())
            if k in inspect.signature(cls).parameters
        }) # type: ignore
        
    name            : str = ''
    symbol          : str = ''
    interval        : int | timedelta | None = None
    column_flags    : Semantics = Semantics.NONE
    
    # Below are values which can be inherited from DatasetConfig.
    # Intervals to resample to
    resample_intervals  : list[str]             = field(default_factory = lambda: [])
    # Indicators to generate data for. Used as input columns by default but may be disabled.
    indicators          : list[IndicatorColumn] = field(default_factory = lambda: [])
    # Other columns to generate, as well as columns from the OHLCV data to include
    include_columns     : list[SourceColumn]    = field(default_factory = lambda: [])
    
    @property
    def include_column_names(self) -> list[str]:
        return [ col_info.name for col_info in self.include_columns ]
    
    @property
    def should_generate_intervals(self):
        has_resamplable_data = \
            ((Semantics.OHLC in self.column_flags) or (Semantics.BIDASK in self.column_flags)) and \
            (Semantics.TIMESTAMP in self.column_flags)
        
        # If resample_intervals isn't falsy and the datasource has resamplable data, generate intervals.
        return self.resample_intervals and has_resamplable_data
    
    @property
    def should_generate_indicators(self):
        has_resamplable_data = \
            ((Semantics.OHLC in self.column_flags) or (Semantics.BIDASK in self.column_flags)) and \
            (Semantics.TIMESTAMP in self.column_flags)
        
        # If resample_intervals isn't falsy and the datasource has resamplable data, generate indicators.
        return self.indicators and has_resamplable_data
    
    def get_default_aggregator(self):
        if Semantics.OHLCV in self.column_flags:
            aggregator = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        elif Semantics.OHLC in self.column_flags:
            aggregator = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            
        # TODO: add bid/ask
        else:
            # Don't generate indicators or intervals if no OHLC or bid/ask data.
            return None
            
        return aggregator
    
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
        self._df = self._retrieve_dataframe(source_json, self.config, force_overwrite)
        self._df = self.resample_data()
    
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
    def df(self):
        return self._df
    
    @property
    def columns(self):
        return self._df.columns
    
    @classmethod
    def get_subclass(cls, name):
        for subclass in cls.__subclasses__():
            if subclass.__name__ == name:
                return subclass
        return None
    
    @classmethod
    def parse_interval(cls, interval_str: str) -> int | timedelta | None:
        if interval_str.lower() == "variable":
            return None
        
        match = re.match(r'([0-9]+)(t|tick|ticks|sec|second|seconds|min|minute|minutes|hr|hour|hours|d|day|days|w|week|weeks)', interval_str)
        if match is None:
            raise ValueError("Malformed interval. Must be tick, timdelta, or 'variable'.")
            
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
        return self._df
    
    def resample_data(self) -> pd.DataFrame:
        """Resamples the standardized dataframe into different intervals and generates indicators. 

        Returns:
            DataFrame: New DataFrame containing resampled data.
        """
        self.verify_df_interval(self.config.column_flags)
        self.verify_df_columns(self.config.column_flags)
        
        new_df = self.resample_and_generate_data(self.config.resample_intervals)
        
        if new_df is not None:
            return new_df
        else:
            return self._df
    
    def get_col_name(self, col, interval):
        return f"{self.config.name}_{col}_{interval}"
    
    def resample_and_generate_data(self, resample_intervals):
        # Get aggregator functions
        aggregator = self.config.get_default_aggregator()
        if aggregator is None:
            return
        
        # Get interval of data
        base_interval = self.config.interval
        
        # Step 1: Input step
        # - Specify columns to draw into resampling and indicator generation step and columns to output/scale.
        column_flags = self.config.column_flags.get_columns()
        
        # Columns marked as present according to datasource semantic flags.
        flag_columns: list[SourceColumn] = [
            col for col in self.config.include_columns if col.name in set(column_flags)
        ]
        # All base data columns - flag columns + indicator/function columns to be generated.
        data_columns: list[SourceColumn|IndicatorColumn] = (
            flag_columns + self.config.indicators
        )
        
        # Columns to make resampled copies of during resampling step, ordered the same way as data_columns.
        column_names_to_resample = set(col.name for col in data_columns if col.name in aggregator.keys())
        columns_to_resample = [ col for col in data_columns if col.name in column_names_to_resample ]
        # Columns to keep in the final DataFrame.
        columns_to_keep = [ col for col in data_columns if col.is_input ]
        # Columns to scale in the final DataFrame.
        columns_to_scale = [ col for col in columns_to_keep if col.is_scaled ]
        
        # If data is fixed-interval:
        if type(base_interval) == timedelta:
            # Index by timestamp
            df = self._df.set_index('timestamp', inplace=False)
                    
            # Step 2: Base indicator step
            # - Generate indicators for the base interval.
            # - This also yields a list of indicator columns to include.
            # - If the semantics are incompatible with all indicators, this will not add any columns.
            resampled_df, ind_columns, ind_infos = self.generate_indicators(df)
            
            # Step 3: Resampling step
            # - If generating different time intervals, resample to different timedeltas.
            # - For each time interval,
            # --- Generate intervals for the time interval.
            # --- 
            if self.config.should_generate_intervals:
                intervals = [base_interval] + resample_intervals
            
                # Get aggregator
                aggregator = self.config.get_default_aggregator()
                if aggregator is None:
                    # If aggregator could not be generated, the columns necessary for resampling are not present.
                    raise ValueError("Failed resampling of Datasource due to missing aggregator columns.") 
            else:
                intervals = [base_interval]
            
            # Resample to multiple intervals and generate indicators for resampled columns.
            for interval in intervals:
                # Resample if not base interval
                if interval != base_interval:
                    resampled_df, old_columns, new_columns = self.resample_fixed(
                        df, base_interval, aggregator, interval, 
                    )
                    
                    # Keep track of scaled columns
                    scaled_columns.update(self.get_col_name(col, interval) for col, info in zip(ind_columns, ind_infos) if info.is_scaled )
                    
                # Add new columns to master df
                df[new_columns] = resampled_df[old_columns]
            
            # Step 4: Rename step
            # - Renames columns to include the base interval and the dataset name for uniqueness.
            # Append '_interval' to column names
            old_columns = columns_to_keep
            new_columns = [ self.get_col_name(col, base_interval) for col in old_columns ]
            df.rename(copy=False)
        
            return df
        
        # If data is variable-interval:
        else:
            raise ValueError("Failed resampling & data generation of Datasource due to malformed base interval.")
            
            
    def resample_fixed(self, df: pd.DataFrame, base_interval: timedelta, aggregator: dict, interval: str):
        """Resamples the internal dataframe from any timestamped datatype (bidask, ohlc, misc)
        into fixed-interval (e.g. 5m ohlc) or fixed-tick (e.g. 9t ohlc or 9-sample aggregate) columns with the given intervals.
        Can resample: <br>
        BID/ASK to: x tick OHLC; or fixed-interval OHLC <br>
        x tick OHLC to: multiple of x tick OHLC; or fixed-interval OHLC <br>
        any OHLC to: any OHLC with greater interval <br>
        anything else to: fixed-interval aggregate data
        """
        td_interval = self.parse_interval(interval)
        columns_to_resample = list(aggregator.keys())
        
        if type(td_interval) == int:
            # Tick resampling
            # TODO: implement tick resampling
            raise ValueError("TODO: Tick resampling not supported")
        elif type(td_interval) == timedelta:
            # Timedelta resampling
            if td_interval <= base_interval:
                raise ValueError("Failed resampling of Datasource due to interval {interval} less than or equal to base interval {base_interval}.")
            
            # Resample and copy new columns from resampled df to new df
            # The usual interval format is compatible with resample().
            resampled_df = df[columns_to_resample].resample(interval).agg(aggregator).dropna() # type: ignore
            
            # Generate indicators for resampling interval (and rename them)
            resampled_df, ind_columns, ind_infos = self.generate_indicators(resampled_df)
            
            # Append '_interval' to column names
            old_columns = columns_to_resample + ind_columns
            new_columns = [ self.get_col_name(col, interval) for col in old_columns ]
            
            return resampled_df, old_columns, new_columns
            
        else:
            raise ValueError("Failed resampling of Datasource due to malformed resampling interval.")
    
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
        ts_col = self._df['timestamp']
        
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
        
        if self._df is None:
            raise ValueError("Failed verification of Datasource due to null DataFrame.")
        
        if not self.check_df_columns(self._df, format_flags):
            raise ValueError(f"Failed verification of Datasource due to missing necessary columns for column flags {self.config.column_flags}. Present columns: {self._df.columns}")
    
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