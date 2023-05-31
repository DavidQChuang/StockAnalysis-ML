from models.Common import StandardModule
# from models.GatedMLP import GatedMLP
from models.SimpleLSTM import SimpleLSTM
# from models.TemporalFusionTransformer import TemporalFusionTransformer

import torch
import torch.backends.mps
from torch import nn
from torch.utils.data import DataLoader

def from_run(run_data) -> StandardModule:
    if 'model' not in run_data:
        raise Exception("'model' cannot be None.")
    if 'model_name' not in run_data:
        raise Exception("'model_name' cannot be None.")
    
    model_json = run_data['model']
    model_name = run_data['model_name']
        
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    
    print(f"> libutil.models:")
    print(f"Using model '{model_name}'.")
    print(f"Using device '{device}'.")
    print()
    
    nn: StandardModule = None
    match model_name:
        # case 'gMLP':
        #     nn = GatedMLP(model)
        case 'LSTM':
            nn = SimpleLSTM(model_json)
        # case 'TFT':
        #     nn = TemporalFusionTransformer(model)
        
    if nn == None:
        raise Exception("Model not found.")
    
    model_json = nn.to(device)
    
    return model_json