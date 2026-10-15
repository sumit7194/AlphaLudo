"""MLX float32 inference port of V17Net (same weights, same outputs).

Fused attention (mx.fast.scaled_dot_product_attention with the edge-bias mask)
and a compiled graph. Verified against the PyTorch model in
v17/tests/test_mlx_model.py — float32 only (no reduced precision anywhere).
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import geometry as g
from .model import DICE_IDX


def _gelu(x):
    return 0.5 * x * (1.0 + mx.erf(x * (1.0 / math.sqrt(2.0))))


def _ln(x, w, b, eps=1e-5):
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * w + b


class MlxV17:
    def __init__(self, torch_model):
        sd = {k: v.detach().float().cpu().numpy() for k, v in torch_model.state_dict().items()}
        self.H = torch_model.n_heads

        def A(n):
            return mx.array(sd[n])

        def L(p):
            return (mx.array(sd[p + ".weight"].T), mx.array(sd[p + ".bias"]))

        self.inp = [L("input_mlp.0"), L("input_mlp.2")]
        self.pos = A("pos_emb.weight")
        self.cls = A("cls")
        etm = torch_model.etm.cpu().numpy()
        self.layers = []
        for i in range(len(torch_model.layers)):
            p = f"layers.{i}."
            eb = sd[p + "attn.edge_bias.weight"]
            self.layers.append({
                "ln1": (A(p + "ln1.weight"), A(p + "ln1.bias")),
                "qkv": L(p + "attn.qkv"), "out": L(p + "attn.out"),
                "ln2": (A(p + "ln2.weight"), A(p + "ln2.bias")),
                "f1": L(p + "ffn.0"), "f2": L(p + "ffn.2"),
                "bias": mx.array(np.transpose(eb[etm], (2, 0, 1))[None]),
            })
        self.fln = (A("final_ln.weight"), A("final_ln.bias"))
        self.pol = [L("policy_mlp.0"), L("policy_mlp.2")]
        self.val = [L("value_mlp.0"), L("value_mlp.2")]
        self.dice_idx = mx.array(np.array(DICE_IDX))
        self._fwd = mx.compile(self._forward)

    def _forward(self, x, mask):
        lin = lambda t, p: t @ p[0] + p[1]
        B = x.shape[0]
        h = lin(_gelu(lin(x, self.inp[0])), self.inp[1]) + self.pos
        h = mx.concatenate([h, mx.broadcast_to(self.cls, (B, 1, h.shape[-1]))], axis=1)
        N, D = h.shape[1], h.shape[2]
        H = self.H
        dh = D // H
        for l in self.layers:
            y = _ln(h, *l["ln1"])
            qkv = lin(y, l["qkv"]).reshape(B, N, 3, H, dh).transpose(2, 0, 3, 1, 4)
            o = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=1.0 / math.sqrt(dh), mask=l["bias"])
            h = h + lin(o.transpose(0, 2, 1, 3).reshape(B, N, D), l["out"])
            h = h + lin(_gelu(lin(_ln(h, *l["ln2"]), l["f1"])), l["f2"])
        h = _ln(h, *self.fln)
        logits = lin(_gelu(lin(h[:, : g.N_CELLS], self.pol[0])), self.pol[1])[..., 0]
        logits = mx.where(mask > 0.5, logits, mx.array(-1e9, dtype=mx.float32))
        v = lin(_gelu(lin(h[:, g.CLS_INDEX], self.val[0])), self.val[1])        # (B, 4)
        corners = x[:, self.dice_idx, :4]                                        # (B, 4, 4)
        active = mx.stack([corners[:, k, k] for k in range(4)], axis=1) >= 1
        v = mx.where(active, v, mx.array(-1e9, dtype=mx.float32))
        return mx.concatenate([mx.softmax(logits, axis=-1), mx.softmax(v, axis=-1)], axis=1)

    def __call__(self, x: np.ndarray, mask: np.ndarray):
        """x: (B, 1125) frames; mask (B, 225). Returns numpy (policy (B,225), value (B,4))."""
        B = x.shape[0]
        out = np.array(self._fwd(mx.array(np.asarray(x, np.float32).reshape(B, g.N_CELLS, 5)),
                                 mx.array(np.asarray(mask, np.float32))))
        return out[:, : g.N_CELLS], out[:, g.N_CELLS:]
