
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Common import ModelConfig
from models.GatedMLP import DropoutLayers, GatedMLPConfig, QKVAttention, SpatialGatingUnit2

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

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
        

class GatedCNNBlock(nn.Module):
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
        
class GatedCNN(nn.Module):
    def __init__(self, model_json):
        super().__init__()
        
        if "gmlp" not in model_json:
            raise Exception("'gmlp' key must be present in model.gmlp parameters.")
        
        gmlp = GatedMLPConfig.from_dict(model_json['gmlp'])
        conf = ModelConfig.from_dict(model_json)
        
        self.feature_count = feature_count = len(conf.columns)
        self.embedding_length = embedding_length = gmlp.embedding_length
        self.seq_len = seq_len = conf.seq_len
        self.output_size = out_seq_len = conf.out_seq_len
        
        d_ffn = conf.hidden_layer_size
        d_model = embedding_length
        d_attn = gmlp.attention_size
        
        L = gmlp.layer_count
        
        # GMLP stack
        layers = [GatedCNNBlock(d_ffn, d_model, seq_len, d_attn, activation=nn.GELU()) for i in range(L)]
        
        # Layers
        # Conv layers
        self.expand = nn.Conv1d(feature_count, embedding_length, 1, bias=False)
        # self.expand = MBConv2d()
        
        # Dense layers
        self.stack = nn.Sequential(*layers) if gmlp.layer_dropout == 0 else DropoutLayers(layers, gmlp.layer_dropout)
        self.unembed = nn.Linear(d_model, out_seq_len)
        
        self.lstm = nn.LSTM(d_model, d_ffn, batch_first=True)
        self.linear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, 1)
        )
        
        self.unproj = nn.Linear(seq_len, 1)

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
        # input_offset = x[:, 0, 0].unsqueeze(-1).clone().detach()
        
        # -- Offset
        # INPUT: x:                     (b, n, f)
        # INPUT: input_offset:          (b, 1)
        # x[:, :, 0] = x[:, :, 0] - input_offset
        
        # -- Depthwise convolution to expand channels
        # INPUT: x:                     (b, n, f)
        # GET:   x:                     (b, d, n)
        x = einops.rearrange(x, "b n f -> b f n")
        x = self.expand(x)
        
        x = einops.rearrange(x, "b d n -> b n d")
        
        # -- FFN
        # INPUT: x:                     (b, n, d)
        # GET:   x = L(x):              (b, n, d)    
        x = self.stack(x)
        
        # -- Unencode
        # INPUT: x:                     (b, n, d)
        #        x = lstm(x):           (b, n, h)
        # GET:   x = linear(x):         (b, n, 1)
        x, hx = self.lstm(x)
        x = self.linear(x)
        
        # -- Unoffset
        # INPUT: x:                     (b, n, 1)
        # INPUT: input_offset:          (b, 1)
        # x = x + (input_offset - output_offset)
        x = self.unproj(x.squeeze(-1))
        # x = x + output_offset
        
        return x