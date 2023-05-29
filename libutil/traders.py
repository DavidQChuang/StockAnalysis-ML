# from traders.AlphaVantage import AlphaVantageDataset

def from_run(run_data):
    if 'data_retriever' not in run_data:
        raise "'data_retriever' cannot be None."
    if 'data_retriever_name' not in run_data:
        raise "'data_retriever_name' cannot be None."
    
    data_retriever = run_data.data_retriever
    data_retriever_name = run_data.data_retriever_name
    
    match data_retriever_name:
        case 'alphavantage':
            return None
            return AlphaVantageDataset(data_retriever)
        case _:
            raise "Model not found."