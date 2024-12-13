import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from .Common import ModelConfig, PytorchModel

class LSTMBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        
        self.lstm = nn.LSTM(size, size)
        self.linear = nn.Linear(size, size)
        self.relu = nn.ReLU()
        self.hx = None
        
    def forward(self, x):
        x, self.hx = self.lstm(x)
        x = self.linear(x)
        x = self.relu(x)
        return x
        
        

class SimpleLSTM(nn.Module):
    def __init__(self, model_json):
        super().__init__()
        
        conf = ModelConfig.from_dict(model_json)
        
        input_size = conf.seq_len
        output_size = conf.out_seq_len
        size = conf.hidden_layer_size
        
        # self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(input_size, size),
            nn.Dropout(conf.dropout_rate),
            
            LSTMBlock(size),
            LSTMBlock(size),
            LSTMBlock(size),
            
            nn.Linear(size, output_size)
        )

    def forward(self, x):
        # x = self.flatten(x)
        input_offset = x[:, 0][:, None]
        
        x = x - input_offset
        x = self.stack(x)
        x = x + input_offset
        return x