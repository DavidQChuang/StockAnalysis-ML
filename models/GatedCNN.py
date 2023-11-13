
import torch
import torch.nn as nn
from models.Common import ModelConfig
from models.GatedMLP import DropoutLayers, GatedMLPBlock, GatedMLPConfig

class MBConv2dUnit(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class MBConv2d(nn.Module):
    def __init__(self,
                 in_channels:int, out_channels:int,
                 expansion_factor=1, kernel_size:int=3, stride:int=1,
                 norm_layer=None):
        super().__init__()
        
        self.stride = stride
        assert stride in [1, 2]
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        ffn_channels = int(round(in_channels*expansion_factor))
        
        self.use_residual = in_channels == out_channels and stride == 1
        
        layers = []
        if expansion_factor != 1:
            layers += [MBConv2dUnit(in_channels, ffn_channels, 1)]
        
        layers += [MBConv2dUnit(ffn_channels, ffn_channels,
                                kernel_size, stride, groups=ffn_channels, norm_layer=norm_layer)]
        layers += [nn.Conv2d(ffn_channels, out_channels, 1, 1, 0, bias=False)]
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)
        
class GatedMLP(nn.Module):
    def __init__(self, model_json):
        super().__init__()
        
        if "gmlp" not in model_json:
            raise Exception("'gmlp' key must be present in model.gmlp parameters.")
        
        gmlp = GatedMLPConfig.from_dict(model_json['gmlp'])
        conf = ModelConfig.from_dict(model_json)
        
        self.feature_count = feature_count = len(conf.columns)
        self.embedding_length = embedding_length = gmlp.embedding_length
        self.seq_len = seq_len = conf.seq_len
        self.output_size = output_size = conf.out_seq_len
        
        d_ffn = conf.hidden_layer_size
        d_model = embedding_length * feature_count
        d_attn = gmlp.attention_size
        
        L = gmlp.layer_count
        
        # GMLP stack
        layers = [GatedMLPBlock(d_ffn, d_model, seq_len, d_attn, activation=nn.GELU()) for i in range(L)]
        
        position_embed = self.get_position_encoding(seq_len, embedding_length)
        self.register_buffer("position_embed", position_embed, True)
        
        # Layers
        self.stack = nn.Sequential(*layers) if gmlp.layer_dropout == 0 else DropoutLayers(layers, gmlp.layer_dropout)
        self.unembed = nn.Linear(d_model, output_size)
        
        # conv_layers = [
        #     nn.Conv2d(feature_count, 32, 3, 2),
        #     MBConv2d(32, 16, 1, 3, 1),
        #     MBConv2d(16, 32, 6, 3, 2),
        #     # MBConv2d(32, 128, 6, 3, 1),
        #     # nn.Conv2d(128, 256, 1, 1),
        #     nn.AvgPool2d(7),
        #     nn.Conv2d(32, 16, 1, 2)
        # ]
        # self.conv = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTM(16, 16, batch_first=True)
        # self.relu = nn.ReLU(inplace=True)
        self.unproj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        
        # self.lstm = nn.LSTM(d_model, d_ffn, batch_first=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.unproj = nn.Linear(seq_len*d_ffn, output_size)

    def forward(self, x):
        # b: batch_size, n: seq_len, f: features, d: embedding_size, o: output_size, h: d_ffn
        # Make sure x is shape (batch_size, seq_len, features)
        # This unsqueezes x from (b, n) to (b, n, 1)
        if len(x.shape) == 1:
            x = x[None, :]
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        
        # Assume 'close' is the 1st column
        input_offset = x[:, 0, 0].unsqueeze(-1).clone().detach()
        
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
        #        x = unproj(x):         (b, n, 16)
        #        x = x.reshape():       (b, n*h)
        # GET:   x = linear(x):         (b, o)
            #    x = x.movedim(2,1)     (b, f*d, n)
            #    x = x.reshape(...)     (b, f, d, n)
            #    x = x.conv(...)        (b, 64, 1, 1)
        # x, hx = self.lstm(x)
        x:torch.Tensor = x.movedim(2,1)
        x = x.reshape((x.shape[0], self.feature_count, self.embedding_length, self.seq_len))
        x = self.conv(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        x, hx = self.lstm(x)
        x = self.unproj(x)
        # x = F.gelu(x)
        # x = x.reshape((x.shape[0], -1))
        # x = x.movedim(1, -1)
        # x = self.unproj(x)
        
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