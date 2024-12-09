from dataclasses import dataclass
import inspect
from sqlite3 import paramstyle

import numpy as np
import pandas as pd
import re

from finta import TA

from datasets.datasources.columns import IndicatorColumn

def get_indicator_name(ind_json=None, ind_conf=None):
    if ind_conf is None:
        ind_conf = IndicatorColumn.from_dict(ind_json)
    elif ind_json is None:
        raise ValueError("Failed to generate indicator name. No json or config was provided.")
    
    # close_ by default, but if 'column' param is present (which defines the column to use for the indicator),
    # then {column}_
    part_a = ind_conf.params['column'] if 'column' in ind_conf.params else 'close'
    # function name in lowercase
    part_b = ind_conf.function.lower()
    # other params in alphabetical order by key
    part_c = '_'.join(value for key,value in sorted(ind_conf.params, key=lambda t:t[0]) if key != 'column')
        
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

        # If no name given, use the function + period as the column name
        if ind_conf.name is None:
            ind_conf.name = get_indicator_name(ind_conf)
        indicator_names.append(ind_conf.name)
        indicator_configs.append(ind_conf)

        df[ind_conf.name] = ind_values

    # Remove values without indicators
    if start_index != 0:
        df = df.iloc[start_index:, :]
        
    return df, indicator_names, indicator_configs