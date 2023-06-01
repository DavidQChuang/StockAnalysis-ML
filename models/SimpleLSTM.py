import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from .Common import PytorchStandardModule

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
        
        

class SimpleLSTM(PytorchStandardModule):
    def __init__(self, model_json, device=None):
        super().__init__(model_json, device)
        
        conf = self.conf
        
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
        input_offset = x[:, 0]
        
        x = x - input_offset[:, None]
        x = self.stack(x)
        x = x + input_offset[:, None]
        return x