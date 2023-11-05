from torch import nn
import torch
import torch.nn.functional as F

from models.GatedMLP import DropoutLayers, GatedMLPBlock, GatedMLPConfig

class GatedMLP_DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hl_size, gmlp=GatedMLPConfig()):
        super().__init__()
        
        feature_count = 1
        embedding_length = gmlp.embedding_length
        seq_len = n_observations
        output_size = n_actions
        
        d_ffn = hl_size
        d_model = embedding_length * feature_count
        d_attn = gmlp.attention_size
        
        L = gmlp.layer_count
        
        # GMLP stack
        layers = [GatedMLPBlock(d_ffn, d_model, seq_len, d_attn, activation=nn.GELU()) for i in range(L)]
        
        position_embed = self.get_position_encoding(seq_len, embedding_length)
        self.register_buffer("position_embed", position_embed, True)
        
        # Layers
        self.stack = nn.Sequential(*layers) if gmlp.layer_dropout == 0 else DropoutLayers(layers, gmlp.layer_count)
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