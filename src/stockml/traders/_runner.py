
import torch
from .Common import StandardTrader
# from stockml.trader.LSTMDQN import LSTMTrader


def from_run(run_data, device=None, use_deepspeed=False, **kwargs):
    if 'trader_name' not in run_data:
        raise ValueError("'trader_name' cannot be None.")
    else:
        if 'trader' not in run_data:
            raise ValueError("'trader' cannot be None if the trader name is given.")
        
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
            raise ValueError("Model not found.")