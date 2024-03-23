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
import models.embeddings as embeddings

@dataclass(frozen=True)    
class GatedMLPConfig:
    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
        
    layer_count: int = 12
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
        self.bias = nn.Parameter(torch.ones(size=(seq_len,)), True)
        self.weight = nn.Parameter(torch.FloatTensor(size=(1, seq_len, seq_len)).uniform_(-init_eps, init_eps), True)
        
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
        
        self.norm = nn.LayerNorm([d_model])
        self.proj_in = nn.Linear(d_model, d_ffn)
        self.activation = activation
        self.sgu = SpatialGatingUnit2(d_ffn, seq_len)
        self.proj_out = nn.Linear(d_ffn // 2, d_model)
        
    def forward(self, x):
        residual = x
        
        if self.attn != None:
            gate_res = self.attn(x)
        else:
            gate_res = None
        
        x = self.norm(x)
        x = self.proj_in(x)
        if self.activation != None:
            x = self.activation(x)
        x = self.sgu(x, gate_res)
        x = self.proj_out(x)
        
        return x + residual
    
    
class DropoutLayers(nn.Module):
    def __init__(self, layers, dropout_rate):
        super().__init__()
        if isinstance(layers, list):
            layers = nn.ModuleList(layers)
        
        self.layers = layers
        self.layer_count = len(layers)
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        if self.training:
            keep_idxs = torch.zeros(self.layer_count).uniform_(0, 1) > self.dropout_rate
            for i, layer in enumerate(self.layers):
                if keep_idxs[i]:
                    x = layer(x)
            
        return x

class GatedMLP(nn.Module):
    def __init__(self, model_json=None, gmlp: GatedMLPConfig|None=None, conf: ModelConfig|None=None):
        super().__init__()
        
        if model_json != None:
            if "gmlp" not in model_json:
                raise Exception("'gmlp' key must be present in model.gmlp parameters.")
            
            gmlp = GatedMLPConfig.from_dict(model_json['gmlp'])
            conf = ModelConfig.from_dict(model_json)
        else:
            gmlp = gmlp if gmlp != None else GatedMLPConfig()
            conf = conf if conf != None else ModelConfig()
        
        self.feature_count = feature_count = len(conf.columns)
        self.embedding_length = embedding_length = gmlp.embedding_length
        self.seq_len = seq_len = conf.seq_len
        self.output_size = out_seq_len = conf.out_seq_len
        
        d_ffn = conf.hidden_layer_size
        d_attn = gmlp.attention_size
        if embedding_length == 0:
            d_model = feature_count
        else:
            d_model = embedding_length * feature_count
            
        L = gmlp.layer_count
        
        # GMLP stack
        layers = [GatedMLPBlock(d_ffn, d_model, seq_len, d_attn, activation=nn.GELU()) for i in range(L)]
        
        if embedding_length != 0:
            position_embed = embeddings.get_position_encoding(seq_len, embedding_length)
            self.register_buffer("position_embed", position_embed, True)
        
        # Layers
        self.stack = nn.Sequential(*layers) if gmlp.layer_dropout == 0 else DropoutLayers(layers, gmlp.layer_dropout)
        # self.unembed = nn.Linear(d_model, out_seq_len)
        
        self.lstm = nn.LSTM(d_model, d_ffn, batch_first=True)
        self.linear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, 1)
        )
        
        self.unproj = nn.Linear(seq_len, out_seq_len)

    def forward(self, x):
        # b: batch_size, n: seq_len, f: features, d: embedding_size, o: output_size, h: d_ffn
        # Make sure x is shape (batch_size, seq_len, features)
        # This unsqueezes x from (n) to (1, n, 1)
        if len(x.shape) == 1:
            x = x[None, :, None]
        if len(x.shape) == 2:
            # This unsqueezes x from (n, f) to (1, n, f)
            if x.shape[1] == self.feature_count:
                x = x.unsqueeze(0)
            # This unsqueezes x from (b, n) to (b, n, 1)
            else:
                x = x.unsqueeze(-1)
        
        # Assume 'close' is the 1st column
        input_offset = x[:, 0, 0].unsqueeze(-1).clone().detach()
        output_offset = x[:, -1, 0].unsqueeze(-1).clone().detach()
        
        # -- Offset
        # INPUT: x:                     (b, n, f)
        # INPUT: input_offset:          (b, 1)
        x[:, :, 0] = x[:, :, 0] - input_offset
        
        # -- Positional encoding
        # INPUT: x:                     (b, n, f)
        # INPUT: positional_encoding:   (n, d)
        #        x.split(1, -1):        list[f]: (b, n, d)
        #        xs[i] * pos_enc:       (b, n, d)
        #        x = torch.stack(xs):   (b, n, f, d)
        #        x = x.reshape():       (b, n, f*d)
        # Split by feature and add positional encoding to each feature datapoint
        if self.embedding_length != 0:
            xs = list(x.split(1, -1))
            for i in range(0, x.shape[-1]):
                xs[i] = xs[i] * self.position_embed
            x = torch.stack(xs, -1)
            x = x.reshape(x.shape[0:2]+(-1,))
        
        # -- FFN
        # INPUT: x:                     (b, n, f*d)
        # GET:   x = L(x):              (b, n, f*d)    
        x = self.stack(x)
        
        # -- Unencode
        # INPUT: x:                     (b, n, f*d)
        #        x = lstm(x):           (b, n, h)
        # GET:   x = linear(x):         (b, n, o)
        x, hx = self.lstm(x)
        x = self.linear(x)
        
        # -- Unoffset
        # INPUT: x:                     (b, n, o)
        # INPUT: input_offset:          (b, o)
        # x = x.squeeze(-1) + (input_offset - output_offset)
        x = self.unproj(x.squeeze(-1))
        x = x + input_offset
        
        return x