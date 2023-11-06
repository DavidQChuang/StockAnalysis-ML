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
        
        if not os.path.exists(file_path):
            raise Exception(f"CSV file `{file_path}` does not exist.")
        
        df = pd.read_csv(file_path)
        super().__init__(df, conf)