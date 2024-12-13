from .Common import TimeSeriesDataset, AdvancedTimeSeriesDataset

def from_run(run_data, **kwargs) -> TimeSeriesDataset:
    if 'dataset' not in run_data:
        raise Exception("'dataset' cannot be None.")
    if 'dataset_name' not in run_data:
        raise Exception("'dataset_name' cannot be None.")
    
    dataset_json = run_data["dataset"]
    
    return AdvancedTimeSeriesDataset(dataset_json)