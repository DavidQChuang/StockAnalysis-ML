from models.Common import PytorchModel, StandardModel, DeepspeedModel
from models.GatedMLP import GatedMLP
from models.GatedCNN import GatedCNN
from models.SimpleLSTM import SimpleLSTM
# from models.TemporalFusionTransformer import TemporalFusionTransformer

import torch
import torch.backends.mps
from torch import nn
from torch.utils.data import DataLoader

import re
from prettytable import PrettyTable

def from_run(run_data, device=None, use_deepspeed=False, **kwargs) -> StandardModel:
    if 'model' not in run_data:
        raise Exception("'model' cannot be None.")
    if 'model_name' not in run_data:
        raise Exception("'model_name' cannot be None.")
    
    model_json = run_data['model']
    model_name:str = run_data['model_name']
        
    if device == None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    
    print(f"> Model loader parameters:")
    print(f"Using model {model_name}.")
    print(f"Using device {device}.")
    
    # If deepspeed use a wrapper
    if model_name.startswith("Deepspeed"):
        use_deepspeed = True
        model_name = re.sub(r"Deepspeed\[([a-zA-Z0-9]+)\]", r"\1", model_name)
    
    if use_deepspeed:
        print(f"Using deepspeed with model {model_name}.")
    
    print()
    
    network = None
    match model_name:
        case 'GatedMLP':
            network = GatedMLP(model_json)
        case 'GatedCNN':
            network = GatedCNN(model_json)
        case 'SimpleLSTM':
            network = SimpleLSTM(model_json)
        # case 'TFT':
        #     nn = TemporalFusionTransformer(model_json, device=device)
        
    if network == None:
        raise Exception("Model not found.")
    
    if isinstance(network, nn.Module):
        if use_deepspeed:
            model = DeepspeedModel(network, model_json, device=device)
        else:
            model = PytorchModel(network, model_json, device=device)
    else:
        raise TypeError("Invalid network type")
    
    return model
        
        