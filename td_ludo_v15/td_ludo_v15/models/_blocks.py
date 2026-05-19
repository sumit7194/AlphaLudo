"""Reusable building blocks for V15's Graph Transformer (and future V16 GNN).

We hand-roll dense attention over ~226 nodes — small enough that the
straightforward `(B, N, N)` attention matrix is cheap (~50K entries per
batch element per head). No PyTorch Geometric dependency.

Components:
    EdgeBiasedAttention: multi-head self-attention with a per-edge-type
        bias added to attention logits. Edge bias is a learned table
        `(num_edge_types, num_heads)` indexed by the static edge-type
        matrix `(N, N)`.
    GTLayer: pre-LN Transformer layer (LN → attn → residual → LN → FFN
        → residual). Standard recipe.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBiasedAttention(nn.Module):
    """Multi-head self-attention with learned per-edge-type bias.

    Forward args:
        x: (B, N, d_model)
        edge_type_matrix: (N, N) int64 — pre-computed at startup; constant
            across batch and across all forward calls. Indexes into the
            per-head edge-bias table.

    The attention logits are computed as:
        logits[b, h, i, j] = (Q[b,h,i] · K[b,h,j]) / sqrt(d_head)
                            + edge_bias[edge_type_matrix[i, j], h]

    Type 0 (NO_EDGE) gets its own bias too — typically learned to 0 or
    slightly negative, meaning "unconnected pairs are softly suppressed."
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_edge_types: int,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Edge bias: per-type, per-head. Initialized to zero so attention
        # starts as standard scaled-dot-product. Model learns the bias.
        self.edge_bias = nn.Embedding(num_edge_types, n_heads)
        nn.init.zeros_(self.edge_bias.weight)
        self.attn_dropout_p = attn_dropout

    def forward(self, x: torch.Tensor, edge_type_matrix: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head
        qkv = self.qkv(x)  # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to (B, H, N, Dh)
        q = q.reshape(B, N, H, Dh).transpose(1, 2)
        k = k.reshape(B, N, H, Dh).transpose(1, 2)
        v = v.reshape(B, N, H, Dh).transpose(1, 2)
        # Attention logits: (B, H, N, N)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
        # Edge bias lookup: edge_type_matrix is (N, N) int → bias is (N, N, H) → permute to (H, N, N)
        bias = self.edge_bias(edge_type_matrix)  # (N, N, H)
        bias = bias.permute(2, 0, 1)             # (H, N, N)
        attn_logits = attn_logits + bias.unsqueeze(0)  # (B, H, N, N)
        attn = F.softmax(attn_logits, dim=-1)
        if self.training and self.attn_dropout_p > 0:
            attn = F.dropout(attn, p=self.attn_dropout_p)
        # Apply attention to values
        out = torch.matmul(attn, v)              # (B, H, N, Dh)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        return out


class GTLayer(nn.Module):
    """One Graph Transformer encoder layer with pre-LN.

    Pre-LN: x = x + Attn(LN(x))
            x = x + FFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        num_edge_types: int,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = EdgeBiasedAttention(
            d_model, n_heads, num_edge_types, attn_dropout=attn_dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout) if ffn_dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor, edge_type_matrix: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), edge_type_matrix)
        x = x + self.ffn(self.ln2(x))
        return x
