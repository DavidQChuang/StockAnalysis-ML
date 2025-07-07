from dataclasses import dataclass, field
from enum import Flag, auto
import inspect

from finta import TA
import numpy as np
import re

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
                part_str = part_str.strip()
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
        
        return [ col.name.lower() for col in expected_columns ]

@dataclass
class SourceColumn:
    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        }) # type: ignore
        
    name      : str  = ''
    
    is_scaled : bool = False
    is_input  : bool = True
    keep_resampled : bool = False

@dataclass
class IndicatorColumn(SourceColumn):
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    # Describes the function to use for this indicator
    function     : str  = ''
    
    # Describes the number of data points to drop from the start of the df
    offset       : int  = 0
    
    # Describes parameters of this indicator. Only used with TA indicators.
    params       : dict[str, str] = field(default_factory = lambda: {})
    
    # TODO: support other cases
    def get_indicator_name(self):
        # close_ by default, but if 'column' param is present (which defines the column to use for the indicator),
        # then {column}_
        part_a = self.params['column'] if 'column' in self.params else 'close'
        # function name in lowercase
        part_b = self.function.lower()
        # other params in alphabetical order by key
        part_c = '_'.join(value for key,value in sorted(self.params, key=lambda t:t[0]) if key != 'column')
            
        return '_'.join([part_a, part_b, part_c])
    

def generate_indicators(df, indicators):
    indicator_names: list[str] = []
    indicator_configs: list[IndicatorColumn] = []
    
    # Get list of functions
    get_attrs = lambda cls: map(lambda name: (getattr(cls, name), name), dir(cls))
    
    ta_functions = {f"TA.{name}": attr for attr, name in get_attrs(TA) if not name.startswith('__')}
    np_functions = {f"np.{name}": attr for attr, name in get_attrs(np) if not name.startswith('__') and callable(attr)}
    
    numpy_regex = re.compile(r'(np.[a-zA-Z_][a-zA-Z0-9_]*+)\(([a-zA-Z_][a-zA-Z0-9_]*)\)')
    
    for ind_json in indicators:
        ind_conf = IndicatorColumn.from_dict(ind_json)
        
        # How many values to remove from the start of the array due to insufficient data points
        values_to_remove = ind_conf.offset
        
        # Function string
        func = ind_conf.function
        params = ind_conf.params if ind_conf.params else {}
        
        if func.startswith("TA."):
            if func not in ta_functions:
                raise ValueError("Failed indicator generation. Indicator function was not found: " + ind_conf.function)
            
            # Call TA function by name
            ind_values = ta_functions[func](*[df], **params)
            
        elif func.startswith("np."):
            if func not in np_functions:
                raise ValueError("Failed indicator generation. Indicator function was not found: " + ind_conf.function)
            
            matches = numpy_regex.match(func)
            if matches != None:
                # First group is numpy function name
                func = matches.group(1)
                # Second group is the name of the column to pass into the function
                param = df[matches.group(2)] 
                
                # Call np function by name
                ind_values = np_functions[func](*[param], **params)
            else:
                raise ValueError("Failed indicator generation. Indicator function began with np but does not match expected format np.func(param): " + ind_conf.function)
        else:
            raise Exception("Invalid indicator function: " + ind_conf.function)
        
        # Set amount of values to remove to the largest removal size
        start_index = max(start_index, values_to_remove)

        # If name is falsy, use an autogenerated indicator name
        if not ind_conf.name:
            ind_conf.name = ind_conf.get_indicator_name()
            
        indicator_names.append(ind_conf.name)
        indicator_configs.append(ind_conf)

        df[ind_conf.name] = ind_values

    # Remove values without indicators
    if start_index != 0:
        df = df.iloc[start_index:, :]
        
    return df, indicator_names, indicator_configs