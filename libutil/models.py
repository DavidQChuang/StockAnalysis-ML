from models.Common import StandardModule
from models.GatedMLP import GatedMLP
from models.SimpleLSTM import SimpleLSTM
# from models.TemporalFusionTransformer import TemporalFusionTransformer

import torch
import torch.backends.mps
from torch import nn
from torch.utils.data import DataLoader

def from_run(run_data, device=None, **kwargs) -> StandardModule:
    if 'model' not in run_data:
        raise Exception("'model' cannot be None.")
    if 'model_name' not in run_data:
        raise Exception("'model_name' cannot be None.")
    
    model_json = run_data['model']
    model_name = run_data['model_name']
        
    if device == None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    
    print(f"> Model loader parameters:")
    print(f"Using model {model_name}.")
    print(f"Using device {device}.")
    print()
    
    model: StandardModule = None
    match model_name:
        case 'GatedMLP':
            model = GatedMLP(model_json, device=device)
        case 'SimpleLSTM':
            model = SimpleLSTM(model_json, device=device)
        # case 'TFT':
        #     nn = TemporalFusionTransformer(model_json, device=device)
        
    if model == None:
        raise Exception("Model not found.")
    
    if isinstance(model, nn.Module):
        model = model.to(device)
    
    return model