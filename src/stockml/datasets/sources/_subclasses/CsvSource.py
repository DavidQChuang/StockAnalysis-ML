
import os
import pandas as pd

from ..Common import Source

class CsvSource(Source):
    def __init__(self, datasource_json, force_overwrite):
        if 'csv' not in datasource_json:
            raise Exception("'csv' key must be present in dataset parameters.")
        query_params = datasource_json['csv']
        
        file_path = query_params['file_path']
        
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
        
        super().__init__(datasource_json, force_overwrite)