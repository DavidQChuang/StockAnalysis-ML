from torch.utils.data.dataset import Dataset, ChainDataset 

import os
import pandas as pd
from datetime import date
from urllib.parse import urlencode

class AlphaVantageDataset(Dataset):
    def __init__(self, data_retriever_json, forceOverwrite=False):
        url = "https://www.alphavantage.co/query?"
        
        if 'alphavantage' not in data_retriever_json:
            raise Exception("'alphavantage' key must be present in run.")
        
        query_params = data_retriever_json['alphavantage']
        
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

        query_string = urlencode(query_params)
        
        dfs = []
        
        # For each slice, download a csv
        for slice in slices:
            slice_str = "" if slice == None else "&slice=%s"%(slice)
            
            url = url + query_string + slice_str
        
            df = self.download_csv(url, self.get_filename(query_params, slice), forceOverwrite)
            dfs.append(df)
            
        self.df = pd.concat(dfs)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self.df[index]
    
    def get_filename(self, query_params, slice):
        def get_param(param):
            return query_params[param] if param in query_params else None
        
        def param_name(param) -> str | None:
            return "" if param == None else str(param)
        
        function = get_param('function')
        ticker = get_param('symbol')
        interval = get_param('interval')
        dt = date.today()
        
        names = {
            "TIME_SERIES_DAILY": "d",
            "TIME_SERIES_DAILY_ADJUSTED": "d",
            "TIME_SERIES_INTRADAY": "i",
            "TIME_SERIES_INTRADAY_EXTENDED": "i",
            "DIGITAL_CURRENCY_DAILY": "dc-d"
        }
        
        if function in names:
            function = names[function]
                
        return 'csv/%s/%s%s-%s%s.csv'%(
            dt,
            function, param_name(ticker),
            param_name(interval), param_name(slice))
        
    def download_csv(self, url: str, fileName: str, forceOverwrite: bool=False):
        # timestamp,open,high,low,close,volume
        if forceOverwrite or not os.path.exists(fileName):
            # print('Loading data from AlphaVantage url : %s' % url)
            df = pd.read_csv(url)
            
            if not 'close' in df.columns and not 'close (USD)' in df.columns:
                print('! Failed loading data from AlphaVantage url %s: '%url)
                print('Columns: ' + df.keys())
                raise FileNotFoundError('Failed to retrieve data from AlphaVantage. Invalid URL %s'%url)
            else:
                if not os.path.isdir('csv'):
                    os.mkdir('csv')
                
                if 'time' in df.columns:
                    df.rename(columns = {'time':'timestamp'}, inplace = True)
                    
                usd_cols_dict = { col:col.removesuffix(' (USD)')
                                 for col in df.columns if col.endswith('(USD)') }
                    
                df.rename(columns = usd_cols_dict, inplace = True)
                df.to_csv(fileName)
        else:
            # print('Loading data from csv, original url : %s' % url)
            df = pd.read_csv(fileName)
            
        return df