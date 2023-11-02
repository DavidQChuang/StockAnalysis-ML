import torch
from traders.Common import StandardTrader
# from traders.LSTMDQN import LSTMTrader


def from_run(run_data, device=None, use_deepspeed=False, **kwargs):
    if 'trader' not in run_data:
        raise "'trader' cannot be None."
    if 'trader_name' not in run_data:
        raise "'trader_name' cannot be None."
        
    if device == None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    
    trader = run_data["trader"]
    trader_name = run_data["trader_name"]
    
    print(f"> Trader loader parameters:")
    print(f"Using trader {trader_name}.")
    print(f"Using device {device}.")
    
    match trader_name:
        case 'none':
            return None
        case 'StandardTrader':
            return StandardTrader(trader, device)
        case _:
            raise "Model not found."