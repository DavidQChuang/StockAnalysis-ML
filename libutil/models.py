from models.GatedMLP import GatedMLP
from models.SimpleLSTM import SimpleLSTM
from models.TemporalFusionTransformer import TemporalFusionTransformer

import torch
import torch.backends.mps

def from_run(run_data):
    if 'model' not in run_data:
        raise "'model' cannot be None."
    if 'model_name' not in run_data:
        raise "'model_name' cannot be None."
    
    model = run_data.model
    model_name = run_data.model_name
    
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else " cpu"
    )
    print(f"> Using {device} device")
    
    nn = None
    match model_name:
        case 'gMLP':
            nn = GatedMLP(**model)
        case 'LSTM':
            nn = SimpleLSTM(**model)
        case 'TFT':
            nn = TemporalFusionTransformer(**model)
        
    if nn == None:
        raise Exception("Model not found.")
    
    
    