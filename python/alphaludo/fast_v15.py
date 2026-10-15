"""Inference-only fast path for V15GraphTransformer (same weights, same math).

The training model hand-rolls attention: it materialises the full
(B, H, 226, 226) logits and re-gathers the edge-bias table on every layer of
every forward. For search (millions of evaluations) this wrapper
  * precomputes each layer's (H, N, N) edge bias once,
  * uses fused F.scaled_dot_product_attention with that bias as attn_mask,
  * optionally runs in float16 (fp16 checked against fp32 on real states).
Outputs match V15GraphTransformer.forward: (masked softmax policy, sigmoid value).
Rebuild it (FastV15(model)) after the source model's weights change.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastV15(nn.Module):
    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.m = model
        self.dtype = dtype
        self.n_heads = model.n_heads
        etm = model.edge_type_matrix
        with torch.no_grad():
            self.biases = [
                layer.attn.edge_bias(etm).permute(2, 0, 1).contiguous().to(dtype)  # (H, N, N)
                for layer in model.layers
            ]
            self.pos = model.pos_emb.weight.detach().to(dtype)
        if dtype != torch.float32:
            self.m = self.m.to(dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor):
        m = self.m
        B = x.shape[0]
        x = x.to(self.dtype).permute(0, 2, 3, 1, 4).reshape(B, 225, -1)
        h = m.input_mlp(x) + self.pos.unsqueeze(0)
        h = torch.cat([h, m.cls_token.to(self.dtype).expand(B, -1, -1)], dim=1)  # (B, 226, D)
        H = self.n_heads
        for layer, bias in zip(m.layers, self.biases):
            a = layer.attn
            y = layer.ln1(h)
            N, D = y.shape[1], y.shape[2]
            q, k, v = a.qkv(y).view(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(0))
            h = h + a.out_proj(o.transpose(1, 2).reshape(B, N, D))
            h = h + layer.ffn(layer.ln2(h))
        h = m.final_ln(h)
        logits = m.policy_mlp(h[:, :225]).squeeze(-1).float()
        logits = logits.masked_fill(legal_mask < 0.5, -1e9)
        value = torch.sigmoid(m.value_mlp(h[:, 225]).squeeze(-1).float())
        return F.softmax(logits, dim=-1), value
