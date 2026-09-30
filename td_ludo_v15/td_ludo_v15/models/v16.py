"""V16 — V15's GraphTransformer with TWO-PROPERTY connections.

The owner's idea, stated verbatim in the original session:

    "currently in a traditional NN we increase or decrease the connection
     strength based on if you want to boost the behaviour or suppress it...
     what if those connections have 2 properties... the scalar can be adjusted
     based on terminal win/loss signal, while the PHASE could change with
     OTHER SPARSE SIGNALS like cuts, or home, or others."

Every bulk weight becomes

    w_eff = a * (1 + tanh(b))

with the two properties owned by DIFFERENT LEARNING SIGNALS:

    a  <- gradient ONLY from the RL objective   (PPO policy + value, win/loss)
    b  <- gradient ONLY from the AUX objective  (sparse tactical events:
                                                 capture-available, in-danger)

Routing is done with two forward passes and opposite detaching, which is exact:
no aux gradient can reach `a`, no RL gradient can reach `b`. Masking after one
backward would be cheaper but `w` depends on both, so the split would only be
approximate.

WHY THIS FILE LIVES HERE AND NOT IN experiments/
------------------------------------------------
The first attempt at this (experiments/twosignal) used a hand-written RL loop
that diverged from the v15.2 pipeline in ~10 independent ways — 100% self-play
instead of a strong-opponent mix, REINFORCE instead of PPO, no GAE, no entropy
bonus, lr 30x too high. After 212k games the model measured BETTER against weak
scripted bots and WORSE head-to-head against the v15.2 champion it was distilled
from (0.380 -> 0.307 over 1,000 games): it had drifted, not improved. None of
that is evidence about the two-property idea; it is evidence about the harness.
So v16 changes exactly ONE thing versus the champion's recipe — the model — and
reuses train_v15_rich.py + V15RichTrainer as the SAME CODE PATH, not a copy.

EXACT-RESTART PROPERTY
----------------------
`b` initialises to zero, so tanh(0) = 0 and w_eff = a EXACTLY. Loading the
champion's own RL checkpoint into `a` therefore produces a network that is
numerically identical to it at step 0 — same trick as the identity-layer
insertion in the G3 depth probe. Every subsequent difference is attributable to
the routing, with no init confound and no separate SL stage to argue about.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game.cells import CLS_INDEX, NUM_BOARD_CELLS
from ..game.graph import EDGE_TYPE_MATRIX, NUM_EDGE_TYPES

MAIN, AUX, JOINT = "main", "aux", "joint"


class TwoPropLinear(nn.Module):
    """w = a * (1 + tanh(b)); `mode` selects which property receives gradient.

    Parameter NAMES are chosen so a V15 checkpoint maps in mechanically:
    `<prefix>.weight` -> `<prefix>.a`, `<prefix>.bias` -> `<prefix>.bias`.
    """

    def __init__(self, d_in: int, d_out: int, two_prop: bool = True):
        super().__init__()
        # Match nn.Linear's default init exactly so a from-scratch v16 is
        # distributed like a from-scratch v15.
        lin = nn.Linear(d_in, d_out)
        self.a = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())
        self.two_prop = two_prop
        if two_prop:
            self.b = nn.Parameter(torch.zeros(d_out, d_in))   # tanh(0)=0 -> w=a
        self.mode = JOINT

    def weight(self) -> torch.Tensor:
        if not self.two_prop:
            return self.a
        if self.mode == MAIN:            # aux property frozen this pass
            return self.a * (1 + torch.tanh(self.b.detach()))
        if self.mode == AUX:             # amplitude frozen this pass
            return self.a.detach() * (1 + torch.tanh(self.b))
        return self.a * (1 + torch.tanh(self.b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)


def set_mode(module: nn.Module, mode: str) -> None:
    for m in module.modules():
        if isinstance(m, TwoPropLinear):
            m.mode = mode


class V16Attention(nn.Module):
    """EdgeBiasedAttention with two-property qkv / out_proj.

    Structurally identical to models/_blocks.EdgeBiasedAttention — same edge
    bias table, same scaling, same head reshape — so behaviour matches v15
    exactly when b == 0.
    """

    def __init__(self, d_model: int, n_heads: int, num_edge_types: int,
                 attn_dropout: float = 0.0, two_prop: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.qkv = TwoPropLinear(d_model, 3 * d_model, two_prop)
        self.out_proj = TwoPropLinear(d_model, d_model, two_prop)
        self.edge_bias = nn.Embedding(num_edge_types, n_heads)
        nn.init.zeros_(self.edge_bias.weight)
        self.attn_dropout_p = attn_dropout

    def forward(self, x: torch.Tensor, edge_type_matrix: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, H, Dh).transpose(1, 2)
        k = k.reshape(B, N, H, Dh).transpose(1, 2)
        v = v.reshape(B, N, H, Dh).transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
        bias = self.edge_bias(edge_type_matrix).permute(2, 0, 1)
        attn_logits = attn_logits + bias.unsqueeze(0)
        attn = F.softmax(attn_logits, dim=-1)
        if self.training and self.attn_dropout_p > 0:
            attn = F.dropout(attn, p=self.attn_dropout_p)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class V16Layer(nn.Module):
    """Pre-LN GT layer. `ffn` keeps v15's Sequential INDICES (0 and 3 are the
    linears, 2 is Dropout/Identity) so state-dict mapping stays mechanical."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 num_edge_types: int, attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.0, two_prop: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = V16Attention(d_model, n_heads, num_edge_types,
                                 attn_dropout=attn_dropout, two_prop=two_prop)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            TwoPropLinear(d_model, ffn_dim, two_prop),
            nn.GELU(),
            nn.Dropout(ffn_dropout) if ffn_dropout > 0 else nn.Identity(),
            TwoPropLinear(ffn_dim, d_model, two_prop),
        )

    def forward(self, x: torch.Tensor, edge_type_matrix: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), edge_type_matrix)
        x = x + self.ffn(self.ln2(x))
        return x


class V16GraphTransformer(nn.Module):
    """V15GraphTransformer with two-property bulk weights + an aux head.

    Only the TRANSFORMER BULK is two-property (qkv, out_proj, ffn.0, ffn.3).
    Embeddings, LayerNorms, edge-bias table, policy head and value head stay
    plain nn.Linear / nn.Embedding — identical to v15 — so the comparison is
    about the connections inside the trunk, not about the readouts.

    forward() returns (policy, value) exactly like v15, so it is a drop-in for
    every caller in the pipeline: the player, the trainer, the bot eval, the
    self-play picker. `want_aux=True` additionally returns the aux logits and
    is used only by the aux backward pass.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 256,
        history_len: int = 1,
        in_features: int | None = None,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        two_prop: bool = True,
        n_aux: int = 2,
    ):
        super().__init__()
        if in_features is None:
            in_features = history_len * 3
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.history_len, self.in_features = history_len, in_features
        self.two_prop, self.n_aux = two_prop, n_aux

        self.input_mlp = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(NUM_BOARD_CELLS, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.register_buffer(
            "edge_type_matrix",
            torch.from_numpy(EDGE_TYPE_MATRIX.astype(np.int64)),
            persistent=False,
        )
        self.layers = nn.ModuleList([
            V16Layer(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim,
                     num_edge_types=NUM_EDGE_TYPES, attn_dropout=attn_dropout,
                     ffn_dropout=ffn_dropout, two_prop=two_prop)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.policy_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3), nn.GELU(),
            nn.Linear(d_model // 3, 1),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3), nn.GELU(),
            nn.Linear(d_model // 3, 1),
        )
        # Aux head — reads the CLS summary, predicts sparse tactical events.
        # NEW relative to v15; it does not feed the trunk, so adding it leaves
        # policy and value bit-identical to the v15 checkpoint at load time.
        self.aux_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3), nn.GELU(),
            nn.Linear(d_model // 3, n_aux),
        )

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(B, NUM_BOARD_CELLS, -1)
        node_emb = self.input_mlp(x.float())
        pos_ids = torch.arange(NUM_BOARD_CELLS, device=x.device)
        node_emb = node_emb + self.pos_emb(pos_ids).unsqueeze(0)
        nodes = torch.cat([node_emb, self.cls_token.expand(B, -1, -1)], dim=1)
        for layer in self.layers:
            nodes = layer(nodes, self.edge_type_matrix)
        return self.final_ln(nodes)

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None,
                want_aux: bool = False):
        nodes = self._trunk(x)
        board_nodes = nodes[:, :NUM_BOARD_CELLS, :]
        cls_node = nodes[:, CLS_INDEX, :]
        policy_logits = self.policy_mlp(board_nodes).squeeze(-1)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(legal_mask.float() < 0.5, -1e9)
        policy = F.softmax(policy_logits, dim=-1)
        value = torch.sigmoid(self.value_mlp(cls_node)).squeeze(-1)
        if want_aux:
            return policy, value, self.aux_mlp(cls_node)
        return policy, value

    # ── V15 interop ────────────────────────────────────────────────────────
    _BULK = ("attn.qkv", "attn.out_proj", "ffn.0", "ffn.3")

    def load_v15_state_dict(self, sd: dict, verbose: bool = True) -> dict:
        """Load a V15GraphTransformer state dict: weight -> a, b stays zero.

        Because b == 0 => w = a, the resulting network is NUMERICALLY IDENTICAL
        to the v15 model it came from. Verified by `assert_matches_v15`.
        Returns a report dict {mapped, copied, missing, unexpected}.
        """
        own = self.state_dict()
        new, mapped, copied = {}, [], []
        for k, v in sd.items():
            if k.endswith(".weight") and any(f".{b}." in f".{k}" for b in self._BULK):
                tgt = k[: -len("weight")] + "a"
                if tgt in own:
                    new[tgt] = v.clone(); mapped.append(f"{k} -> {tgt}")
                    continue
            if k in own and own[k].shape == v.shape:
                new[k] = v.clone(); copied.append(k)
        missing = [k for k in own if k not in new]
        unexpected = [k for k in sd if k not in own and
                      not any(m.startswith(k + " ") for m in mapped)]
        # b stays at its zero init; aux_mlp keeps its random init.
        merged = dict(own); merged.update(new)
        self.load_state_dict(merged, strict=True)
        if verbose:
            print(f"[v16] mapped {len(mapped)} bulk weights -> `a`, "
                  f"copied {len(copied)} tensors verbatim, "
                  f"{len([m for m in missing if m.endswith('.b')])} `b` left at zero, "
                  f"aux head randomly initialised")
        return {"mapped": mapped, "copied": copied,
                "missing": missing, "unexpected": unexpected}

    @torch.no_grad()
    def assert_matches_v15(self, v15_model: nn.Module, x: torch.Tensor,
                           legal_mask: torch.Tensor, tol: float = 1e-5) -> float:
        """Gate: with b == 0 the policy must equal the v15 model's, exactly.

        Run this before every v16 launch. If it fails, the state-dict mapping
        is wrong and any result afterwards would be attributed to routing when
        it is actually an init bug.
        """
        was = self.training
        self.eval(); v15_model.eval()
        set_mode(self, JOINT)
        p16, v16 = self(x, legal_mask)
        p15, v15 = v15_model(x, legal_mask)
        dp = float((p16 - p15).abs().max())
        dv = float((v16 - v15).abs().max())
        if was:
            self.train()
        assert dp < tol and dv < tol, (
            f"v16 does NOT match v15 at init: max|dpolicy|={dp:.3e} "
            f"max|dvalue|={dv:.3e} (tol {tol}). State-dict mapping is broken.")
        return max(dp, dv)

    def gate_stats(self) -> Optional[dict]:
        """Did the second property actually move, or stay at its init?"""
        if not self.two_prop:
            return None
        g = torch.cat([torch.tanh(m.b.detach()).flatten()
                       for m in self.modules() if isinstance(m, TwoPropLinear)])
        return {"gate_abs_mean": float(g.abs().mean()),
                "gate_std": float(g.std()),
                "gate_max_abs": float(g.abs().max()),
                "frac_moved_gt_0.01": float((g.abs() > 0.01).float().mean())}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
