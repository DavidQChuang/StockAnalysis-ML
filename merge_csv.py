import glob
import pandas as pd
from tqdm import tqdm

files = glob.glob("csv/2023-06-01/iTQQQ-5m*.csv", recursive=True)
files = [ 'csv/iTQQQ-5m[2021-06-11,2023-05-31].csv', 'csv/2023-06-13/iTQQQ-5m-y1m1.csv', 'csv/2023-06-23/iTQQQ-5m.csv']

big_df = pd.read_csv(files[0])
big_df = big_df.set_index('timestamp')

for file in tqdm(files[1:]):
    df = pd.read_csv(file)
    big_df = big_df.combine_first(df.set_index('timestamp'))
           
big_df.sort_index(inplace=True)
big_df.to_csv("csv/iTQQQ-5m.csv")