"""MLX inference port of V15GraphTransformer (history_len=1), same weights.

Apple's MLX runs the attention through a fused kernel
(mx.fast.scaled_dot_product_attention with the additive edge-bias mask) and
compiles the whole forward graph. Outputs match the PyTorch model's
forward(): (masked softmax policy (B, 225), sigmoid value (B,)) — verified
against PyTorch on real positions before use.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def _gelu(x):
    return 0.5 * x * (1.0 + mx.erf(x * (1.0 / math.sqrt(2.0))))


def _ln(x, w, b, eps=1e-5):
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * w + b


def _lin(x, p):
    return x @ p[0] + p[1]


class MlxV15:
    """dtype: compute dtype of matmuls/attention. mixed=True keeps the residual
    stream, LayerNorms, softmax and heads in float32 and casts only the
    matmul / attention inputs to `dtype` (accuracy close to fp32, speed
    close to fp16)."""

    def __init__(self, torch_model, dtype=mx.float32, mixed: bool = False):
        sd = {k: v.detach().float().cpu().numpy() for k, v in torch_model.state_dict().items()}
        self.dtype = dtype
        self.mixed = mixed
        f32 = mx.float32
        self.H = torch_model.n_heads

        res_dt = f32 if mixed else dtype  # dtype of residual-stream tensors

        def A(name, dt=None):
            return mx.array(sd[name]).astype(dt or res_dt)

        def L(prefix, dt=None):  # torch Linear weight (out, in) -> (in, out)
            d = dt or dtype
            return (mx.array(sd[prefix + ".weight"].T).astype(d), mx.array(sd[prefix + ".bias"]).astype(d))

        self.inp = [L("input_mlp.0"), L("input_mlp.2")]
        self.pos = A("pos_emb.weight")
        self.cls = A("cls_token")
        etm = torch_model.edge_type_matrix.cpu().numpy()
        self.layers = []
        for i in range(torch_model.n_layers):
            p = f"layers.{i}."
            eb = sd[p + "attn.edge_bias.weight"]                     # (types, H)
            bias = np.transpose(eb[etm], (2, 0, 1))[None]            # (1, H, N, N)
            self.layers.append({
                "ln1": (A(p + "ln1.weight"), A(p + "ln1.bias")),
                "qkv": L(p + "attn.qkv"), "out": L(p + "attn.out_proj"),
                "ln2": (A(p + "ln2.weight"), A(p + "ln2.bias")),
                "f1": L(p + "ffn.0"), "f2": L(p + "ffn.3"),
                "bias": mx.array(bias).astype(dtype),
            })
        self.fln = (A("final_ln.weight"), A("final_ln.bias"))
        head_dt = f32 if mixed else dtype
        self.pol = [L("policy_mlp.0", head_dt), L("policy_mlp.2", head_dt)]
        self.val = [L("value_mlp.0", head_dt), L("value_mlp.2", head_dt)]
        self._fwd = mx.compile(self._forward)

    def _forward(self, x, mask):
        B = x.shape[0]
        H = self.H
        lo = self.dtype                                   # matmul / attention dtype
        rs = mx.float32 if self.mixed else self.dtype     # residual-stream dtype

        def lin(t, p):  # matmul in `lo`, result back in the residual dtype
            return (t.astype(lo) @ p[0] + p[1]).astype(rs)

        h = lin(_gelu(lin(x.astype(rs), self.inp[0])), self.inp[1]) + self.pos                 # (B, 225, D)
        h = mx.concatenate([h, mx.broadcast_to(self.cls, (B, 1, h.shape[-1]))], axis=1)       # (B, 226, D)
        N, D = h.shape[1], h.shape[2]
        dh = D // H
        scale = 1.0 / math.sqrt(dh)
        for l in self.layers:
            y = _ln(h, *l["ln1"])
            qkv = (y.astype(lo) @ l["qkv"][0] + l["qkv"][1]).reshape(B, N, 3, H, dh).transpose(2, 0, 3, 1, 4)
            o = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=scale, mask=l["bias"])
            h = h + lin(o.transpose(0, 2, 1, 3).reshape(B, N, D), l["out"])
            h = h + lin(_gelu(lin(_ln(h, *l["ln2"]), l["f1"])), l["f2"])
        h = _ln(h, *self.fln)
        hd = mx.float32 if self.mixed else lo
        logits = _lin(_gelu(_lin(h[:, :225].astype(hd), self.pol[0])), self.pol[1])[..., 0].astype(mx.float32)
        logits = mx.where(mask > 0.5, logits, mx.array(-1e9, dtype=mx.float32))
        value = mx.sigmoid(_lin(_gelu(_lin(h[:, 225].astype(hd), self.val[0])), self.val[1])[..., 0].astype(mx.float32))
        # One output array: converting it to numpy runs the whole graph once.
        return mx.concatenate([mx.softmax(logits, axis=-1), value[:, None]], axis=1)

    def __call__(self, x: np.ndarray, mask: np.ndarray):
        """x: (B, 675) frames (15*15*3, history 1); mask: (B, 225).
        Returns numpy (policy (B, 225), value (B,))."""
        B = x.shape[0]
        xm = mx.array(np.asarray(x, dtype=np.float32).reshape(B, 225, 3))
        out = np.array(self._fwd(xm, mx.array(np.asarray(mask, dtype=np.float32))))
        return out[:, :225], out[:, 225]
