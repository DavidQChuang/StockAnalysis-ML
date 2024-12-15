from .Common import ModelConfig, PytorchModel, StandardModel, DeepspeedModel
from ._subclasses import GatedMLP, GatedCNN, SimpleLSTM
# from stockml.models.TemporalFusionTransformer import TemporalFusionTransformer

from stockml.datasets import TimeSeriesDataset

import torch
import torch.backends.mps
from torch import nn

import re

def from_run(run_data, dataset: TimeSeriesDataset, device=None, use_deepspeed=False, **kwargs) -> StandardModel:
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
    
    conf = ModelConfig.from_dict(model_json)
    
    network = None
    match model_name:
        case 'GatedMLP':
            gmlp = GatedMLP.get_config(model_json)
            network = GatedMLP(conf, gmlp)
        case 'GatedCNN':
            gmlp = GatedCNN.get_config(model_json)
            network = GatedCNN(conf, gmlp)
        case 'SimpleLSTM':
            network = SimpleLSTM(conf)
        # case 'TFT':
        #     nn = TemporalFusionTransformer(model_json, device=device)
        
    if network == None:
        raise Exception("Model not found.")
    
    if isinstance(network, nn.Module):
        if use_deepspeed:
            model = DeepspeedModel(network, dataset.columns, dataset.scaled_columns, model_json, device=device)
        else:
            model = PytorchModel(network, dataset.columns, dataset.scaled_columns, model_json, device=device)
    else:
        raise TypeError("Invalid network type")
    
    return model
        
        