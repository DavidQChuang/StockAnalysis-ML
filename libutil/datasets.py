from datasets.AlphaVantage import AlphaVantageDataset
from datasets.Common import TimeSeriesDataset

def from_run(run_data, **kwargs) -> TimeSeriesDataset:
    if 'dataset' not in run_data:
        raise Exception("'dataset' cannot be None.")
    if 'dataset_name' not in run_data:
        raise Exception("'dataset_name' cannot be None.")
    
    dataset_json = run_data["dataset"]
    dataset_name = run_data["dataset_name"]
    
    match dataset_name:
        case 'alphavantage':
            return AlphaVantageDataset(dataset_json)
        case _:
            raise "Model not found."