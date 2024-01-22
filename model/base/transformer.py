import torch
from torch import nn
import math 
from torch.autograd import Variable

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class ChannelReduce(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out 


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, out_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, ChannelReduce(dim, out_dim, dropout = dropout)),
            PreNorm(out_dim, FeedForward(out_dim, mlp_dim, dropout = dropout))
        ]))
        for _ in range(depth - 1):
            self.layers.append(nn.ModuleList([
                PreNorm(out_dim, Attention(out_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(out_dim, ChannelReduce(out_dim, out_dim, dropout = dropout)),
                PreNorm(out_dim, FeedForward(out_dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, dr, ff in self.layers:
            x = attn(x) + dr(x)
            x = ff(x) + x
        return x