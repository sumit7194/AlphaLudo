"""V13.3 — Temporal CNN + Transformer over last K turns.

Hypothesis: every architecture so far (V12.2 attn-CNN, V13.2 deep-CNN,
V14_scalar DeepSets) is **stateless and single-frame**. They all
converged to the 80-83% plateau. This is *partly* evidence about
architecture variety, but also evidence about a shared limitation:
none of them see what the opponent did in recent turns. V13.3 tests
that limitation directly by stacking the last K frames as a temporal
context, fed to a small transformer.

Differences from V13.2:
  - Backbone: 4 ResBlocks × 64 channels per turn (~150K params, vs
    V13.2's 10×128 ~3M). Smaller because each forward processes K
    frames.
  - Temporal: 2-layer transformer over K=8 turns of GAP-pooled CNN
    embeddings, with sinusoidal positional encoding + a "history mask"
    for early-game turns where K-1 prior frames don't exist yet.
  - Output: same 3-head shape as V13.x line (policy + win_prob +
    moves_remaining). Heads attach to the LAST turn's transformer
    output (representing "now").

Total params at default sizing: ~350-400K (close to V14_scalar's 226K).

Inference uses a per-game cache of the K-1 most recent turn embeddings
so each new move only computes one new CNN forward + a transformer
pass. Training computes all K frames per state for simplicity (cost
manageable at 4×64 backbone size).

Forward shape (training mode):
  x       : (B, K, 17, 15, 15) float32  — last K V17-encoded frames,
                                          oldest first; pad zeros
                                          + history_mask for missing
  history_mask: (B, K) bool                — True where the frame is real;
                                          False where padded (early game)
  legal_mask  : (B, 4) float32           — 1.0 where action legal at "now"
  Returns (policy, win_prob, moves_remaining), heads attached to the
  last turn's transformer output.

Forward shape (inference mode, single-step with cache):
  current_frame : (B, 17, 15, 15)
  cached_embeds : (B, K-1, hidden)
  cached_mask   : (B, K-1) bool
  legal_mask    : (B, 4)
  Returns same 3-tuple.

NOTE: This is the *non-cached* simpler implementation for the SL
distillation phase. Cache is only used at play/eval time once we
hook V13.3 into the play server.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Same shape as V13.x ResBlock but configurable width."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


def _sinusoidal_positional_encoding(K: int, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal positional encoding, length K."""
    pe = torch.zeros(K, dim)
    pos = torch.arange(0, K, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                    * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (K, dim)


# ── Main model ─────────────────────────────────────────────────────────────
class V133Temporal(nn.Module):
    """V13.3 — temporal CNN + transformer over K turns."""
    OWN_TOKEN_CHANNELS = (0, 1, 2, 3)

    def __init__(
        self,
        history_k: int = 8,
        in_channels: int = 17,
        cnn_blocks: int = 4,
        cnn_channels: int = 64,
        # Transformer over per-turn embeddings
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        # Heads
        head_hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.history_k = history_k
        self.in_channels = in_channels
        self.cnn_channels = cnn_channels
        self.d_model = d_model

        # Per-turn CNN backbone
        self.conv_input = nn.Conv2d(in_channels, cnn_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(cnn_channels)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(cnn_channels) for _ in range(cnn_blocks)]
        )

        # Project per-turn GAP feature to d_model (in case cnn_channels != d_model).
        if cnn_channels == d_model:
            self.proj_in = nn.Identity()
        else:
            self.proj_in = nn.Linear(cnn_channels, d_model)

        # Sinusoidal positional encoding for K turns.
        self.register_buffer(
            "pos_enc", _sinusoidal_positional_encoding(history_k, d_model)
        )

        # Transformer encoder over K turns (causal not strictly needed since
        # we only ever predict on the LAST position, but we use no mask here
        # — full attention over the K context window).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads attached to the LAST turn's transformer output.
        # Policy: per-token slot (4 logits) — V13.2 used per-token spatial
        # extraction, but here we just produce 4 logits from the temporal
        # embedding. Mask later.
        self.policy_fc = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 4),
        )
        self.win_fc = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        self.moves_fc = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    # ── Backbone ────────────────────────────────────────────────────────
    def _per_turn_cnn(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (N, in_channels, 15, 15) → (N, cnn_channels) GAP feature."""
        out = F.relu(self.bn_input(self.conv_input(frames)))
        for block in self.res_blocks:
            out = block(out)
        # GAP → (N, cnn_channels)
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    @staticmethod
    def _apply_legal_mask(logits: torch.Tensor, legal_mask: torch.Tensor | None):
        if legal_mask is None:
            return logits
        all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
        logits = logits.masked_fill(~legal_mask.bool(), float("-inf"))
        if all_illegal.any():
            logits = torch.where(
                all_illegal.expand_as(logits),
                torch.zeros_like(logits),
                logits,
            )
        return logits

    # ── Forward (training, batched K-frame histories) ───────────────────
    def forward(
        self,
        x: torch.Tensor,                     # (B, K, in_ch, 15, 15)
        legal_mask: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,  # (B, K), True = real frame
    ):
        B, K, C, H, W = x.shape
        assert K == self.history_k, f"expected K={self.history_k}, got {K}"
        assert C == self.in_channels

        # CNN per turn: flatten to (B*K, C, H, W) → (B*K, cnn_channels)
        flat = x.reshape(B * K, C, H, W)
        per_turn = self._per_turn_cnn(flat)                    # (B*K, cnn_channels)
        per_turn = self.proj_in(per_turn).reshape(B, K, self.d_model)
        per_turn = per_turn + self.pos_enc.unsqueeze(0)         # (B, K, d_model)

        # Build src_key_padding_mask: True where IGNORE (i.e. NOT real frame)
        if history_mask is not None:
            src_pad = ~history_mask.bool()
        else:
            src_pad = None

        # Transformer over K
        attended = self.transformer(per_turn, src_key_padding_mask=src_pad)  # (B, K, d_model)

        # Take the LAST turn's representation as "now"
        now = attended[:, -1, :]                                  # (B, d_model)

        # Heads
        policy_logits = self.policy_fc(now)                       # (B, 4)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        win_prob = torch.sigmoid(self.win_fc(now)).squeeze(-1)    # (B,)
        moves = F.softplus(self.moves_fc(now)).squeeze(-1)        # (B,)

        return policy, win_prob, moves

    def forward_policy_only(self, x, legal_mask=None, history_mask=None):
        """RL-time interface (mirrors MinimalCNN14): pre-softmax legal-masked
        policy logits."""
        B, K, C, H, W = x.shape
        flat = x.reshape(B * K, C, H, W)
        per_turn = self._per_turn_cnn(flat)
        per_turn = self.proj_in(per_turn).reshape(B, K, self.d_model)
        per_turn = per_turn + self.pos_enc.unsqueeze(0)
        src_pad = ~history_mask.bool() if history_mask is not None else None
        attended = self.transformer(per_turn, src_key_padding_mask=src_pad)
        now = attended[:, -1, :]
        policy_logits = self.policy_fc(now)
        return self._apply_legal_mask(policy_logits, legal_mask)

    # ── Inference cache helpers (for play server, future use) ──────────
    @torch.no_grad()
    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single frame into its d_model embedding (no positional
        encoding applied here — that's positional-dependent, applied later).

        frame: (B, in_channels, 15, 15)
        Returns: (B, d_model)
        """
        feat = self._per_turn_cnn(frame)
        return self.proj_in(feat)

    @torch.no_grad()
    def forward_with_cache(
        self,
        cached_embeds: torch.Tensor,    # (B, K-1, d_model) — oldest first
        cached_mask: torch.Tensor,      # (B, K-1) bool
        new_frame_embed: torch.Tensor,  # (B, d_model) — already projected
        legal_mask: torch.Tensor | None = None,
    ):
        """Inference path used during play. Caller maintains the cache;
        here we just compose the K-window and run transformer + heads."""
        B = new_frame_embed.size(0)
        K = self.history_k

        new_embed = new_frame_embed.unsqueeze(1)           # (B, 1, d_model)
        full_embeds = torch.cat([cached_embeds, new_embed], dim=1)  # (B, K, d_model)

        new_mask = torch.ones(B, 1, dtype=torch.bool, device=cached_mask.device)
        full_mask = torch.cat([cached_mask, new_mask], dim=1)        # (B, K)

        full_embeds = full_embeds + self.pos_enc.unsqueeze(0)
        src_pad = ~full_mask
        attended = self.transformer(full_embeds, src_key_padding_mask=src_pad)
        now = attended[:, -1, :]

        policy_logits = self.policy_fc(now)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        win_prob = torch.sigmoid(self.win_fc(now)).squeeze(-1)
        moves = F.softplus(self.moves_fc(now)).squeeze(-1)

        return policy, win_prob, moves

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== V13.3 V133Temporal ===")
    m = V133Temporal()
    print(f"params: {m.count_parameters():,}")
    print(f"  cnn_channels={m.cnn_channels} d_model={m.d_model} K={m.history_k}")

    # Dummy training-style forward
    B, K = 2, 8
    x = torch.randn(B, K, 17, 15, 15)
    history_mask = torch.tensor([
        [False, False, False, True, True, True, True, True],   # 5 real frames
        [True] * K,                                              # all real
    ])
    legal = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.float)
    p, w, mv = m(x, legal, history_mask)
    print(f"  policy: {tuple(p.shape)}, sum={p.sum(dim=1).tolist()}")
    print(f"  win_prob: {tuple(w.shape)}, vals={w.tolist()}")
    print(f"  moves: {tuple(mv.shape)}")

    # Cached-inference smoke
    cached = torch.randn(B, K - 1, m.d_model)
    cached_mask = torch.ones(B, K - 1, dtype=torch.bool)
    new_emb = m.encode_frame(torch.randn(B, 17, 15, 15))
    p2, w2, mv2 = m.forward_with_cache(cached, cached_mask, new_emb, legal)
    print(f"\ncached-inference forward:")
    print(f"  policy: {tuple(p2.shape)}, sum={p2.sum(dim=1).tolist()}")
