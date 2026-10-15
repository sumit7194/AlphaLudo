"""V17 network — unified 2-player / 4-player GraphTransformer.

Same backbone as the v15.2 champion (d=128, 4 layers, 4 heads, ffn 256, one
frame), with:
  * 5 input features per cell (ludo_core::unified encoding, relative seats),
  * the V17 board graph (v17/graph.py),
  * a 4-way value head: which relative seat wins, softmax over the ACTIVE seats
    only (read from the input's dice corners), so in a 2-player game it is
    exactly P(I win) vs P(opponent wins).

`forward_logits` returns logits for training (policy: illegal cells -1e9;
value: inactive seats -1e9). `forward` returns probabilities for play.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as g
from .graph import EDGE_TYPES, N_EDGE_TYPES

IN_FEATURES = 5
DICE_IDX = [g.idx(*c) for c in g.DICE_CELLS]
NEG = -1e9


class EdgeBiasedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Embedding(N_EDGE_TYPES, n_heads)
        nn.init.zeros_(self.edge_bias.weight)

    def forward(self, x: torch.Tensor, etm: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        logits = logits + self.edge_bias(etm).permute(2, 0, 1).unsqueeze(0)
        o = torch.softmax(logits, dim=-1) @ v
        return self.out(o.transpose(1, 2).reshape(B, N, D))


class Layer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = EdgeBiasedAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(), nn.Linear(ffn, d_model))

    def forward(self, x, etm):
        x = x + self.attn(self.ln1(x), etm)
        return x + self.ffn(self.ln2(x))


class V17Net(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, ffn: int = 256):
        super().__init__()
        self.cfg = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, ffn=ffn)
        self.n_heads = n_heads
        self.input_mlp = nn.Sequential(nn.Linear(IN_FEATURES, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos_emb = nn.Embedding(g.N_CELLS, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.layers = nn.ModuleList([Layer(d_model, n_heads, ffn) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.policy_mlp = nn.Sequential(nn.Linear(d_model, d_model // 3), nn.GELU(), nn.Linear(d_model // 3, 1))
        self.value_mlp = nn.Sequential(nn.Linear(d_model, d_model // 3), nn.GELU(), nn.Linear(d_model // 3, 4))
        self.register_buffer("etm", torch.from_numpy(EDGE_TYPES), persistent=False)
        self.register_buffer("dice_idx", torch.tensor(DICE_IDX), persistent=False)

    @staticmethod
    def frames(x: torch.Tensor) -> torch.Tensor:
        """(B, 1125) or (B, 225, 5) of any dtype -> (B, 225, 5) float32."""
        return x.reshape(x.shape[0], g.N_CELLS, IN_FEATURES).float()

    def active_seats(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4) bool: relative seat k is in the game iff its dice corner f[k] >= 1."""
        corners = x[:, self.dice_idx, :4]                      # (B, 4 corners, 4 features)
        return torch.diagonal(corners, dim1=1, dim2=2) >= 1    # corner k, feature k

    def forward_logits(self, x: torch.Tensor, legal_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.frames(x)
        B = x.shape[0]
        h = self.input_mlp(x) + self.pos_emb.weight.unsqueeze(0)
        h = torch.cat([h, self.cls.expand(B, -1, -1)], dim=1)
        for layer in self.layers:
            h = layer(h, self.etm)
        h = self.final_ln(h)
        policy = self.policy_mlp(h[:, : g.N_CELLS]).squeeze(-1)
        policy = policy.masked_fill(legal_mask < 0.5, NEG)
        value = self.value_mlp(h[:, g.CLS_INDEX])
        value = value.masked_fill(~self.active_seats(x), NEG)
        return policy, value

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p, v = self.forward_logits(x, legal_mask)
        return F.softmax(p, dim=-1), F.softmax(v, dim=-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
