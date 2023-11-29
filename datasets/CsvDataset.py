from tqdm import tqdm
from .Common import DatasetConfig, AdvancedTimeSeriesDataset

import os
import re
import pandas as pd
from datetime import date
from urllib.parse import urlencode

class CsvDataset(AdvancedTimeSeriesDataset):
    def __init__(self, dataset_json):
        conf = DatasetConfig.from_dict(dataset_json)
        
        if 'file_path' not in dataset_json:
            raise Exception("'file_path]' key must be present in dataset parameters.")
        
        file_path = dataset_json['file_path']
        
        if os.path.isdir(file_path):
            dfs = []
            
            files = [f for f in os.listdir('.') if f.endswith('.py')]
            for f in files:
                dfs.append(pd.read_csv(f))
                
            df = pd.concat(dfs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by="timestamp")

        elif os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
        else:
            raise Exception(f"CSV file `{file_path}` does not exist.")
        
        super().__init__(df, conf)