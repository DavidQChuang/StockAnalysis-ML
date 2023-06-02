import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataclasses import dataclass
import inspect

from .Common import PytorchStandardModule
from .SimpleLSTM import LSTMBlock

class SpatialGatingUnit(nn.Module):
    def __init__(self, seq_len, init_eps = 1e-3):
        """
        Args:
            d_ffn (int): dimensionality of the internal feed-forward part (hidden layer size)
            d_model (int): dimensionality (d) of the input (embedding size)
            seq_len (int): length of the token sequence; window size (n)
        """
        # (seq_len, d_model)x(d_model, d_ffn) -> (seq_len, d_ffn)
        # (seq_len, d_ffn) -> (seq_len, d_ffn // 2), (seq_len, d_ffn // 2)
        # (seq_len, seq_len)x(seq_len, d_ffn // 2) -> (seq_len, d_ffn // 2)
        super().__init__()
        
        self.bias = nn.Parameter(torch.ones(size=(seq_len,)))
        self.kernel = nn.Parameter(torch.zeros(size=(1, seq_len, seq_len)))
        # self.kernel = nn.Parameter(torch.FloatTensor(size=(1, seq_len, seq_len)).uniform_(-init_eps, init_eps))
        
    def forward(self, x):
        residual, gate = torch.tensor_split(x, 2, dim=-1)
        
        gate = torch.matmul(self.kernel, gate)
        gate = torch.transpose(gate, 2, 1)
        gate = gate + self.bias[None, :]
        gate = torch.transpose(gate, 2, 1)
            
        return residual * gate

class GatedMLPBlock(nn.Module):
    def __init__(self, d_ffn, d_model, seq_len, activation=None):
        """
        Args:
            d_ffn (int): dimensionality of the internal feed-forward part (hidden layer size)
            d_model (int): dimensionality (d) of the input (embedding size)
            seq_len (int): length of the token sequence; window size (n)
            See 'Pay Attention to MLPs' (arxiv:2105.08050)
        """
        # (seq_len, d_model)x(d_model, d_ffn) -> (seq_len, d_ffn)
        # (seq_len, d_ffn) -> (seq_len, d_ffn // 2), (seq_len, d_ffn // 2)
        # (seq_len, seq_len)x(seq_len, d_ffn // 2) -> (seq_len, d_ffn // 2)
        super().__init__()
        
        self.activation = activation
        
        self.dropout = nn.Dropout()
        self.normalize = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.asgu = SpatialGatingUnit(seq_len)
        self.linear2 = nn.Linear(d_ffn // 2, d_model)
        
    def forward(self, x):
        shortcut = x
        x = self.dropout(x)
        x = self.normalize(x)
        x = self.linear1(x)
        
        if(self.activation is not None):
            x = self.activation(x)
        
        x = self.asgu(x)
        x = self.linear2(x)
        
        return x + shortcut

@dataclass(frozen=True)    
class GatedMLPConfig:
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    layer_count: int = 6
    encoding_length: int = 32
    

class GatedMLP(PytorchStandardModule):
    def __init__(self, model_json, device=None):
        super().__init__(model_json, device)
        
        if "gmlp" not in model_json:
            raise Exception("'gmlp' key must be present in model.gmlp parameters.")
        
        self.conf_gmlp = GatedMLPConfig.from_dict(model_json['gmlp'])
        
        conf = self.conf
        
        seq_len = conf.seq_len
        output_size = conf.out_seq_len
        
        d_ffn = conf.hidden_layer_size
        d_model = self.conf_gmlp.encoding_length
        
        L = self.conf_gmlp.layer_count
        
        # GMLP stack
        layers = [GatedMLPBlock(d_ffn, d_model, seq_len, activation=nn.GELU()) for i in range(L)]
        
        self.positional_encoding = self.get_position_encoding(seq_len, d_model)
        
        # Layers
        self.stack = nn.Sequential(*layers)
        self.lstm = nn.LSTM(d_model, d_ffn)
        self.unproj1 = nn.Linear(d_ffn, output_size)
        self.unproj2 = nn.Linear(seq_len, 1)
        
        if device == 'cuda':
            self.positional_encoding = self.positional_encoding.cuda()

    def forward(self, x):
        # x = self.flatten(x)
        input_offset = x[:, 0][:, None]
        
        x = x - input_offset
        
        # Make sure x is shape (batch_size, seq_len, features)
        # This unsqueezes x if it is (batch_size, seq_len)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # -- Positional encoding
        # INPUT: x: (batch_size, seq_len, features, 1), pos_enc: (None, seq_len, d_model)
        # x = x * pos_enc: (batch_size, seq_len, features, d_model)
        # x = x.flatten:   (batch_size, seq_len, d_model * features)
        # GET: (batch_size, seq_len, d_model * features)
        x = x.unsqueeze(-1)
        x = x * self.positional_encoding[:, None]
        x = x.flatten(2)
        
        # -- FFN
        x = self.stack(x)
        x, hx = self.lstm(x)
        
        # Unproject
        x = self.unproj1(x).squeeze()
        x = self.unproj2(x)
        x = x + input_offset
        
        return x
        
    def get_position_encoding(self, seq_len, d, n=10000):
        P = torch.zeros((seq_len, d), dtype=torch.float32)
        for k in range(seq_len):
            for i in torch.arange(d // 2):
                denominator = torch.pow(n, 2*i/d)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P