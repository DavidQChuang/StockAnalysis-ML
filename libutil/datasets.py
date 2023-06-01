from torch.utils.data.dataset import Dataset

from datasets.AlphaVantage import AlphaVantageDataset

def from_run(run_data, **kwargs) -> Dataset:
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