import os
import torch
import einops
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataclasses import dataclass
import inspect

from .Common import ModelConfig, PytorchModel
from .SimpleLSTM import LSTMBlock

@dataclass(frozen=True)    
class GatedMLPConfig:
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    layer_count: int = 6
    layer_dropout: float = 0
    embedding_length: int = 32
    attention_size: int = 64
    
class QKVAttention(nn.Module):
    def __init__(self, d_in, d_out, d_ffn):
        super().__init__()
        self.scale = d_ffn ** -0.5

        self.to_qkv = nn.Linear(d_in, d_ffn * 3, bias = False)
        self.to_out = nn.Linear(d_ffn, d_out)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        
        sim = torch.einsum('bnd,bmd->bnm', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = torch.einsum('bnm,bmd->bnd', attn, v)
        
        return self.to_out(out)
    
class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len, init_eps = 1e-3):
        """
        Args:
            seq_len (int): length of the token sequence; window size (n)
        """
        super().__init__()
        
        # self.norm = nn.LayerNorm(d_ffn // 2)
        self.bias = nn.Parameter(torch.ones(size=(seq_len,)))
        self.weight = nn.Parameter(torch.FloatTensor(size=(1, seq_len, seq_len)).uniform_(-init_eps, init_eps))
        
    def forward(self, x, gate_res=None):
        weight, bias = self.weight, self.bias
        
        residual, gate = x.chunk(2, dim=-1)
        # gate = self.norm(gate)
        
        gate = torch.matmul(weight, gate)
        gate = torch.transpose(gate, 2, 1)
        gate = gate + bias[None, :]
        gate = torch.transpose(gate, 2, 1)
        
        if gate_res is not None:
            gate += gate_res
            
        return residual * gate
    
class SpatialGatingUnit2(nn.Module):
    def __init__(self, d_ffn, seq_len, heads = 4, init_eps = 1e-3):
        """
        Args:
            seq_len (int): length of the token sequence; window size (n)
        """
        super().__init__()
        
        self.heads = heads
        
        # self.norm = nn.LayerNorm(d_ffn // 2)
        self.bias = nn.Parameter(torch.ones(size=(heads, seq_len,)))
        self.weight = nn.Parameter(torch.FloatTensor(size=(heads, seq_len, seq_len)).uniform_(-init_eps, init_eps))
        
    def forward(self, x, gate_res=None):
        weight, bias = self.weight, self.bias
        
        residual, gate = x.chunk(2, dim=-1)
        # gate = self.norm(gate)
        
        gate = einops.rearrange(gate, 'b n (h d) -> b h n d', h = self.heads)
        gate = torch.einsum('b h n d, h m n -> b h m d', gate, weight)
        gate = gate + einops.rearrange(bias, 'h n -> () h n ()')
        gate = einops.rearrange(gate, 'b h n d -> b n (h d)')
        
        if gate_res is not None:
            gate += gate_res
            
        return residual * gate

class GatedMLPBlock(nn.Module):
    def __init__(self, d_ffn, d_model, seq_len, d_attn=0, activation=None):
        """
        Args:
            d_ffn (int): dimensionality of the internal feed-forward part (hidden layer size)
            d_model (int): dimensionality (d) of the input (embedding size)
            seq_len (int): length of the token sequence; window size (n)
            See 'Pay Attention to MLPs' (arxiv:2105.08050)
        """
        super().__init__()
        
        if d_attn > 0:
            self.attn = QKVAttention(d_model, d_ffn // 2, d_attn)
        else:
            self.attn = None
        
        self.proj_in = nn.Sequential(
            # nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ffn),
            activation or nn.Identity()
        )
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.proj_out = nn.Linear(d_ffn // 2, d_model)
        
    def forward(self, x):
        residual = x
        
        if self.attn is not None:
            gate_res = self.attn(x)
        else:
            gate_res = None
        
        x = self.proj_in(x)
        x = self.sgu(x, gate_res)
        x = self.proj_out(x)
        
        return x + residual
    
    
class DropoutLayers(nn.Module):
    def __init__(self, module_list, dropout_rate):
        if isinstance(module_list, list):
            module_list = nn.ModuleList(module_list)
        
        self.layers = module_list
        self.layer_count = len(module_list)
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        if self.training:
            keep_idxs = torch.zeros(self.layer_count).uniform_(0, 1) > self.dropout_rate
            for i, layer in enumerate(self.module_list):
                if keep_idxs[i]:
                    x = layer(x)
            
        return x

class GatedMLP(nn.Module):
    def __init__(self, model_json):
        super().__init__()
        
        if "gmlp" not in model_json:
            raise Exception("'gmlp' key must be present in model.gmlp parameters.")
        
        gmlp = GatedMLPConfig.from_dict(model_json['gmlp'])
        conf = ModelConfig.from_dict(model_json)
        
        feature_count = 1
        embedding_length = gmlp.embedding_length
        seq_len = conf.seq_len
        output_size = conf.out_seq_len
        
        d_ffn = conf.hidden_layer_size
        d_model = embedding_length * feature_count
        d_attn = gmlp.attention_size
        
        L = gmlp.layer_count
        
        # GMLP stack
        layers = [GatedMLPBlock(d_ffn, d_model, seq_len, d_attn, activation=nn.GELU()) for i in range(L)]
        
        position_embed = self.get_position_encoding(seq_len, embedding_length)
        self.register_buffer("position_embed", position_embed, True)
        
        # Layers
        self.stack = nn.Sequential(*layers) if gmlp.layer_dropout == 0 else DropoutLayers(layers)
        self.unembed = nn.Linear(d_model, output_size)
        self.lstm = nn.LSTM(d_model, d_ffn, batch_first=True)
        self.unproj = nn.Linear(seq_len*d_ffn, output_size)

    def forward(self, x):
        # b: batch_size, n: seq_len, f: features, d: embedding_size, o: output_size, h: d_ffn
        
        # Make sure x is shape (batch_size, seq_len, features)
        # This unsqueezes x from (b, n) to (b, n, 1)
        if len(x.shape) == 1:
            x = x[None, :]
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        
        input_offset = x[:, 0]
        
        # -- Offset
        # INPUT: x:                     (b, n, f)
        # INPUT: x[:, 0][:, None]:      (b, 1)
        x = x - input_offset[:, None]
        
        # -- Positional encoding
        # INPUT: x:                     (b, n, f)
        # INPUT: positional_encoding:   (n, d)
        #        pos_enc[None, :]:      (1, n, d)
        #        x = x * pos_enc:       (b, n, f, d)
        #        x = x.reshape():       (b, n, f*d)
        x = x * self.position_embed[None, :]
        x = x.reshape(x.shape[0:2]+(-1,))
        
        # -- FFN
        # INPUT: x:                     (b, n, f*d)
        # GET:   x = L(x):              (b, n, f*d)    
        x = self.stack(x)
        
        # -- Unencode
        # INPUT: x:                     (b, n, f*d)
        #        x = lstm(x):           (b, n, h)
        #        x = x.reshape():       (b, n*h)
        # GET:   x = linear(x):         (b, o)
        x, hx = self.lstm(x)
        x = F.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = self.unproj(x)
        
        # -- Unoffset
        # INPUT: x:                     (b, o)
        # INPUT: input_offset:          (b, 1)
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