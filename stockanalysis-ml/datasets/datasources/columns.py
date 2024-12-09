from dataclasses import dataclass
import inspect


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
    params       : dict[str, str] = {}
    
    is_input     : bool = False
    is_scaled    : bool = True