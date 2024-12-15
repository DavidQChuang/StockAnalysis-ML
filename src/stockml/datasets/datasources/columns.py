from dataclasses import dataclass, field
from enum import Flag, auto
import inspect

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
        
        return [ col.name.lower() for col, _ in expected_columns ] # type: ignore

@dataclass
class DatasourceColumn:
    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        }) # type: ignore
        
    name      : str  = ''
    is_scaled : bool = False
    is_input  : bool = True

@dataclass
class IndicatorColumn:
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
    function     : str  = ''
    
    offset       : int  = 0
    params       : dict[str, str] = field(default_factory = lambda: {})
    
    is_input     : bool = False
    is_scaled    : bool = True