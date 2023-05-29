from datasets.AlphaVantage import AlphaVantageDataset

def from_run(run_data):
    if 'data_retriever' not in run_data:
        raise "'data_retriever' cannot be None."
    if 'data_retriever_name' not in run_data:
        raise "'data_retriever_name' cannot be None."
    
    dataset = run_data.dataset
    dataset_name = run_data.dataset_name
    
    match dataset_name:
        case 'alphavantage':
            return AlphaVantageDataset(dataset)
        case _:
            raise "Model not found."