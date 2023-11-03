from tqdm import tqdm

from datasets.indicators import get_series_ema, get_series_macd, get_series_sma
from .Common import DatasetConfig, IndicatorConfig, TimeSeriesDataset

import os
import re
import pandas as pd
from datetime import date
from urllib.parse import urlencode

class AlphaVantageDataset(TimeSeriesDataset):
    def __init__(self, dataset_json):
        self.conf = DatasetConfig.from_dict(dataset_json)
        df = self.get_dataframe(dataset_json)
        
        super().__init__(df, conf=self.conf)
    
    def get_dataframe(self, dataset_json, forceOverwrite=False):
        url = "https://www.alphavantage.co/query?"
        
        if 'alphavantage' not in dataset_json:
            raise Exception("'alphavantage' key must be present in dataset parameters.")
        
        query_params = dataset_json['alphavantage']
        
        if 'apikey' not in query_params:
            raise Exception("'apikey' key must be present in dataset.alphavantage parameters.")
        
        # If multiple slices, get the slices
        if 'slices' in query_params:
            if isinstance(query_params['slices'], list):
                slices = query_params['slices']
                del query_params['slices']
            else:
                slices = [ query_params['slices'] ]
                del query_params['slices']
        else:
            slices = [ None ]

        url += urlencode(query_params)
        
        dfs = []
        
        # For each slice, download a csv
        for slice in tqdm(slices, "Downloading from AlphaVantage", ncols=80):
            slice_str = "" if slice == None else "&slice=%s"%(slice)
            
            new_url = url + slice_str
        
            df = self.download_csv(new_url, self.get_filename(query_params, slice), forceOverwrite)
            dfs.append(df)
            
        df = pd.concat(dfs, ignore_index=True)
        
        # Calculate indicators
        start_index = 0
        if self.conf.indicators != None:
            print("Loading indicators " + ','.join(map(lambda x: x['function'], self.conf.indicators)))
            for indicator in self.conf.indicators:
                ind_conf = IndicatorConfig.from_dict(indicator)
                ind_values = [0] * ind_conf.period
                
                # How many values to remove from the start of the arraydue to insufficient data points
                start_index = max(start_index, ind_conf.period)
                close_values = df['close']
                # Whether or not this columns should be scaled with 'close' or left unscaled.
                scale_with_close = True
                
                match ind_conf.function:
                    case 'SMA':
                        ind_values = get_series_sma(ind_conf.period, close_values)
                        
                    case 'EMA':
                        ind_values = get_series_ema(ind_conf.period, close_values)
                        
                    case 'MACD':
                        ind_values = get_series_macd(ind_conf.period, ind_conf.period2, close_values)
                        scale_with_close = False
                            
                    case _:
                        raise Exception("Invalid indicator name: " + ind_conf.function)
                
                col_name = 'close_' + ind_conf.function.lower() + str(ind_conf.period)
                if not scale_with_close:
                    col_name = 'unscaled_' + col_name

                df[col_name] = ind_values
        
        # Remove values without indicators
        df = df.iloc[start_index:, :]
        
        print(df.iloc[0:5, :])
        print()
        
        return df
    
    def get_filename(self, query_params, slice):
        def get_param(param):
            return query_params[param] if param in query_params else ""
        
        def param_name(param):
            return "-"+param if param else ""
        
        function = get_param('function')
        ticker = get_param('symbol')
        interval = get_param('interval')
        dt = date.today()
        
        functions = {
            "TIME_SERIES_DAILY": "d",
            "TIME_SERIES_DAILY_ADJUSTED": "d",
            "TIME_SERIES_INTRADAY": "i",
            "TIME_SERIES_INTRADAY_EXTENDED": "i",
            "DIGITAL_CURRENCY_DAILY": "dc-d"
        }
        
        interval = re.sub(r"([0-9]+)min", r"\1m", interval)
        slice = re.sub(r"year([0-9]+)month([0-9]+)", r"y\1m\2", slice or "")
        
        if function in functions:
            function = functions[function]
                
        return 'csv/%s/%s%s%s%s.csv'%(
            dt,
            function, str(ticker),
            param_name(interval), param_name(slice))
        
    def download_csv(self, url: str, file_name: str, force_overwrite: bool=False):
        # timestamp,open,high,low,close,volume
        if force_overwrite or not os.path.exists(file_name):
            # print('Loading data from AlphaVantage url : %s' % url)
            df = pd.read_csv(url)
            
            if not 'close' in df.columns and not 'close (USD)' in df.columns:
                print('! Failed loading data from AlphaVantage url %s: '%url)
                print('Columns: ' + str(df.columns.values))
                raise FileNotFoundError('Failed to retrieve data from AlphaVantage. You may have exceeded the free 5/min limit. %s'%url)
            else:
                if not os.path.isdir('csv'):
                    os.mkdir('csv')
                    
                idx = file_name.rfind('/')
                dir_name = file_name[:idx]
                    
                if not os.path.isdir(dir_name):
                    os.mkdir(dir_name)
                    
                # Standardize column names 
                
                if 'time' in df.columns:
                    df.rename(columns = {'time':'timestamp'}, inplace = True)
                    
                usd_cols_dict = { col:col.removesuffix(' (USD)')
                                 for col in df.columns if col.endswith('(USD)') }
                    
                df.rename(columns = usd_cols_dict, inplace = True)
                df.to_csv(file_name)
        else:
            # print('Loading data from csv, original url : %s' % url)
            df = pd.read_csv(file_name)
            
        return df