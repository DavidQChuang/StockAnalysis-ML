from models.Common import StandardModule, DeepspeedWrapper
from models.GatedMLP import GatedMLP
from models.SimpleLSTM import SimpleLSTM
# from models.TemporalFusionTransformer import TemporalFusionTransformer

import torch
import torch.backends.mps
from torch import nn
from torch.utils.data import DataLoader

import re
from prettytable import PrettyTable

def from_run(run_data, device=None, use_deepspeed=False, **kwargs) -> StandardModule:
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
    
    model: StandardModule = None
    match model_name:
        case 'GatedMLP':
            model = GatedMLP(model_json, device=device)
        case 'SimpleLSTM':
            model = SimpleLSTM(model_json, device=device)
        # case 'TFT':
        #     nn = TemporalFusionTransformer(model_json, device=device)
    count_parameters(model)
        
    if model == None:
        raise Exception("Model not found.")
    
    if isinstance(model, nn.Module):
        model = model.to(device)
        
    if use_deepspeed:
        model = DeepspeedWrapper(model, model_json, device)
    
    return model
        
        
def count_parameters(model):
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        # table.add_row([name, params])
        total_params+=params
        
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params