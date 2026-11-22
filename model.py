"""Decoder-only transformer built from scratch, with four positional encodings.

pos_type codes used everywhere in this project (and in the checkpoint file names):
    "PE"   = learned absolute positional embedding
    "ROPE" = rotary positional embedding (RoPE)
    "ALBI" = ALiBi (attention with linear biases)
    "NOPE" = no positional encoding (NoPE)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.encode = nn.Embedding(max_len, d_model)

    def forward(self, x):
        T = x.shape[1]
        position = torch.arange(0, T, device=x.device)
        pos = self.encode(position)
        return x + pos


class AlibiEncoding(nn.Module):
    def __init__(self, max_len, number_of_heads):
        super().__init__()
        heads = torch.arange(1, number_of_heads + 1)
        slope = 2 ** ((-8 * heads) / number_of_heads)
        positions = torch.arange(max_len)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        bias = -slope.view(number_of_heads, 1, 1) * distance
        self.register_buffer('bias', bias)

    def forward(self, T):
        return self.bias[:, :T, :T]


class RopeEncoding(nn.Module):
    def __init__(self, head_dim, max_len):
        super().__init__()
        even_dims = torch.arange(0, head_dim, 2, dtype=torch.float32)
        speed = 10000 ** (-even_dims / head_dim)
        position = torch.arange(0, max_len, dtype=torch.float32)
        angles = torch.outer(position, speed)
        self.register_buffer('cos', torch.cos(angles))
        self.register_buffer('sin', torch.sin(angles))

    def forward(self, x):
        cos = self.cos[:x.shape[2]]
        sin = self.sin[:x.shape[2]]
        first = x[..., 0::2]
        second = x[..., 1::2]
        new_first = first * cos - second * sin
        new_second = first * sin + second * cos
        return torch.stack([new_first, new_second], dim=-1).flatten(-2)


class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len, pos_type):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        self.pos_type = pos_type
        self.d_k = d_model // num_heads
        if pos_type == "ROPE":
            self.rope = RopeEncoding(self.d_k, max_len)
        if pos_type == "ALBI":
            # the attribute name "albi" is kept so the saved checkpoints still load
            self.albi = AlibiEncoding(max_len, num_heads)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
        # used by the analysis scripts
        self.last_weights = None    # attention pattern of the last forward pass
        self.last_heads_out = None  # each head's output of the last forward pass
        self.head_mean = None       # average head output, used for mean ablation
        self.head_ablate = None     # index of the head to ablate (None = no ablation)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        if self.pos_type == "ROPE":
            q = self.rope(q)
            k = self.rope(k)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if self.pos_type == "ALBI":
            scores = scores + self.albi(T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights.detach()

        heads_out = weights @ v
        self.last_heads_out = heads_out.detach()
        if self.head_ablate is not None:
            h = self.head_ablate
            if self.head_mean is not None:
                heads_out[:, h] = self.head_mean[h, :T]  # mean ablation
            else:
                heads_out[:, h] = 0                      # zero ablation (fallback)
        merged = heads_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(merged), weights


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.linear_2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(x)))


class ResidualConnections(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class DecoderBlock(nn.Module):
    def __init__(self, attention_layer, feedforward_layer, d_model):
        super().__init__()
        self.attention_layer = attention_layer
        self.feedforward_layer = feedforward_layer
        self.residual_connections = nn.ModuleList([ResidualConnections(d_model) for _ in range(2)])

    def forward(self, x):
        x = self.residual_connections[0](x, lambda y: self.attention_layer(y)[0])
        x = self.residual_connections[1](x, self.feedforward_layer)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, max_len, pos_type, block, embed, pos, norm, out):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        self.pos_type = pos_type
        self.block = nn.ModuleList(block)
        self.embed = embed
        self.pos = pos
        self.norm = norm
        self.out = out

    def forward(self, x):
        x = self.embed(x)
        if self.pos_type == "PE":
            x = self.pos(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        return self.out(x)


def build_transformer(vocab_size, d_model, num_heads, max_len, pos_type, n_layers):
    embed = InputEmbeddings(d_model, vocab_size)
    norm = LayerNorm(d_model)
    pos = LearnedPositionalEncoding(d_model, max_len) if pos_type == "PE" else None
    blocks = []
    for _ in range(n_layers):
        attn = MultiHeadAttention(d_model, num_heads, max_len, pos_type)
        feedforward = FeedForward(d_model)
        blocks.append(DecoderBlock(attn, feedforward, d_model))
    out = nn.Linear(d_model, vocab_size)
    return Transformer(d_model=d_model, num_heads=num_heads, max_len=max_len,
                       pos_type=pos_type, block=blocks, embed=embed,
                       pos=pos, norm=norm, out=out)
