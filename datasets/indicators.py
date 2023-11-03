import pandas as pd
from tqdm import tqdm

def init_indicator(period, series):
    series_size = len(series)
    ind_values = [None]*period
    
    return (series_size, ind_values)

def get_single_sma(index, period, series):
    return series[index-period:index].sum() / period

def get_series_sma(period, series):
    (series_size, ind_values) = init_indicator(period, series)
    
    print("Processing " + str(series_size) + " values")
    for i in tqdm(range(period, series_size), ncols=80):
        ind_values.append(get_single_sma(i, period, series))
    return ind_values
        
def get_series_ema(period, series):
    (series_size, ind_values) = init_indicator(period, series)
    ind_values.append(get_single_sma(period, period, series))
    
    mult = 2/(1+period)
    
    print("Processing " + str(series_size) + " values")
    for i in tqdm(range(period+1, series_size), ncols=80):
        ind_values.append(series[i]*mult + ind_values[-1]*(1-mult))
    return ind_values
        
def get_series_macd(period1, period2, series):
    (series_size, ind_values) = init_indicator(period1, series)
    
    prev_ema1 = get_single_sma(period1, period1, series)
    prev_ema2 = get_single_sma(period2, period2, series)
    ind_values = [None] * period1 + [prev_ema2 - prev_ema1]
    
    mult1 = 2/(1+period1)
    mult2 = 2/(1+period2)
    
    print("Processing " + str(series_size) + " values")
    for i in tqdm(range(period2+1, series_size), ncols=80):
        ema2 = series[i]*mult2 + prev_ema2*(1-mult2)
        prev_ema2 = ema2
            
        if i > period1:
            ema1 = series[i]*mult1 +  prev_ema1*(1-mult1)
            prev_ema1 = ema1
            
            ind_values.append(ema2 - ema1)
    return ind_values
            