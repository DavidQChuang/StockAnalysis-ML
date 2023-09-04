import torch
from torch import nn

from models.SimpleLSTM import LSTMBlock

class LSTMTrader(nn.Module):
    def __init__(self, trader_json, device=None):
        super().__init__(trader_json, device)
        
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
        input_offset = x[:, 0][:, None]
        
        x = x - input_offset
        x = self.stack(x)
        x = x + input_offset
        return x