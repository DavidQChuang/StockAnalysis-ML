from stockml.vprint import vprint
from .Common import TimeSeriesDataset, MultisourceTimeSeriesDataset

def from_run(run_data, verbosity=1, **kwargs) -> TimeSeriesDataset:
    if 'dataset' not in run_data:
        raise Exception("'dataset' cannot be None.")
    
    dataset_json = run_data["dataset"]
    
    vprint(verbosity, 2, "--- Dataset JSON: \n", dataset_json, "\n---")
    
    return MultisourceTimeSeriesDataset(dataset_json, verbosity)