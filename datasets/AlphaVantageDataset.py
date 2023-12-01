import calendar
import pytz
from tqdm import tqdm

from .Common import DatasetConfig, IndicatorConfig, AdvancedTimeSeriesDataset

import os
import re
import pandas as pd
from datetime import date, datetime, timedelta
from urllib.parse import urlencode

class AlphaVantageDataset(AdvancedTimeSeriesDataset):
    def __init__(self, dataset_json):
        df = self.get_dataframe(dataset_json)
        
        super().__init__(df, conf=DatasetConfig.from_dict(dataset_json))
        
    
    def get_dataframe(self, dataset_json, forceOverwrite=False):
        
        if 'alphavantage' not in dataset_json:
            raise Exception("'alphavantage' key must be present in dataset parameters.")
        
        query_params = dataset_json['alphavantage']
        
        # If multiple months, get the months
        if 'months' in query_params:
            if isinstance(query_params['months'], list):
                months = query_params['months']
                del query_params['months']
            else:
                months = query_params['months']
                del query_params['months']
        else:
            months = [ None ]
            
        if 'dir' in query_params:
            dfs = []
            
            filenames = [ self.get_filename(query_params, month, query_params['dir']) for month in months ]
            for file in filenames:
                dfs.append(pd.read_csv(file))
        else:
            dfs = self.download_files(query_params, months, forceOverwrite)
                
        df = pd.concat(dfs, ignore_index=True)
        df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by="timestamp")
        
        df.to_csv("csv/test.csv")
        
        return df
    
    def download_files(self, query_params, months=[None], forceOverwrite=False):
        if 'apikey' not in query_params:
            raise Exception("'apikey' key must be present in dataset.alphavantage parameters.")

        url = "https://www.alphavantage.co/query?"
        url += urlencode(query_params)
        
        dfs = []
        
        # Get time in EST
        current_datetime = datetime.now(pytz.timezone("America/New_York"))
        current_date = current_datetime.date()
        
        # If the day hasn't ended yet, AlphaVantage hasn't updated for the current day.
        if current_datetime.hour < 16:
            current_date -= timedelta(days=1)
            
        if type(months) == str:
            if months[0] == '-':
                first_month = int(months)
                months = [ i for i in range(first_month, 1) ]
                print(months)
            else:
                raise SyntaxError("Invalid month string. Must be -[months].")
        
        # For each slice, download a csv
        for slice in tqdm(months, "Downloading from AlphaVantage", ncols=80):
            
            if slice == None:
                month_str = self.get_month(0, current_date)
                slice_str = ""
            else:
                month_str = self.get_month(int(slice), current_date)
                slice_str = "&month=" + month_str
            
            new_url = url + slice_str
        
            print(new_url)
            df = self.download_csv(new_url,
                    self.get_filename(query_params, month_str, dir="monthly"), forceOverwrite)
            dfs.append(df)
            
        return dfs
    
    def get_month(self, offset: int, current_date: date):
        month = current_date.month
        year = current_date.year
        
        if offset != 0:
            while offset != 0:
                if offset < 0:
                    # Subtract one month for each offset
                    month -= 1
                    if month <= 0:
                        year -= 1
                        month = 12
                    
                    offset += 1
                else:
                    # Add one month for each offset
                    month += 1
                    if month >= 13:
                        year += 1
                        month = 1
                    
                    offset -= 1
                
            return "%d-%02d"%(year, month)
        else: 
            last_day = calendar.monthrange(year, month)[1]
            
            # If this is the last day, use the normal string
            if current_date.day == last_day:
                return "%d-%02d"%(year, month)
            # Else add the day
            else:
                return "%d-%02d-%02d"%(year, month, current_date.day)
    
    def get_filename(self, query_params, slice, dir):
        def get_param(param):
            return query_params[param] if param in query_params else ""
        
        def param_name(param):
            return "-"+param if param else ""
        
        function = get_param('function')
        ticker = get_param('symbol')
        interval = get_param('interval')
        
        functions = {
            "TIME_SERIES_DAILY": "d",
            "TIME_SERIES_DAILY_ADJUSTED": "d",
            "TIME_SERIES_INTRADAY": "i",
            "TIME_SERIES_INTRADAY_EXTENDED": "i",
            "DIGITAL_CURRENCY_DAILY": "dc-d"
        }
        
        if 'adjusted' not in query_params or query_params['adjusted'] == True:
            slice += "adj"
        
        interval = re.sub(r"([0-9]+)min", r"\1m", interval)
        
        if function in functions:
            function = functions[function]
                
        return 'csv/%s/%s%s%s%s.csv'%(
            dir,
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