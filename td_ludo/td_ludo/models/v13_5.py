"""V13.5 — Token-symmetric CNN.

Key changes vs V13.2 (`MinimalCNN14`):
  - Encoder: 13ch V18 (token-permutation-invariant) instead of 17ch V17.
  - Output: 4 logits indexed by *canonical rank* (most-advanced own-token
    position first), not by token-ID. Truly permutation-equivariant under
    token-ID swaps.

Architecture is otherwise V13.2-style: pure CNN trunk (ResBlocks × channels),
GAP for value/moves heads, per-rank spatial extraction via einsum for the
policy head — analogous to V13.2's per-token extraction but rank-indexed.

POC defaults
------------
6 ResBlocks × 96 channels ≈ 1.05M params (vs V13.2's 3.0M at 10×128).
Moderate size: small enough for fast Mac MPS POC, big enough to absorb
V13.2's policy. Scaling-up path matches V13.2's geometry on VM if POC
shows promise.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v18_symmetric import V18_CHANNELS
from td_ludo.game.rank_mapping import (
    HOME_POS,
    MAX_RANK_SLOTS,
    state_to_rank_mapping,
)


# Canonical own-home cell (used when a rank corresponds to the home position).
# All own home cells in V14_minimal are in the 2×2 corner block; we collapse
# them to (2, 2) for the rank mask. The choice is arbitrary but stable.
_OWN_HOME_RANK_CELL = (2, 2)


class _ResBlock(nn.Module):
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


class V135Symmetric(nn.Module):
    """13ch V18 encoder → CNN → rank-indexed policy + value + moves heads."""

    def __init__(
        self,
        num_res_blocks: int = 6,
        num_channels: int = 96,
        in_channels: int = V18_CHANNELS,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.in_channels = in_channels

        self.conv_input = nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList(
            [_ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head — applied per rank after einsum extraction
        self.policy_fc1 = nn.Linear(num_channels, head_hidden)
        self.policy_fc2 = nn.Linear(head_hidden, 1)

        # Value + moves heads — global pool
        self.win_fc1 = nn.Linear(num_channels, head_hidden)
        self.win_fc2 = nn.Linear(head_hidden, 1)
        self.moves_fc1 = nn.Linear(num_channels, head_hidden)
        self.moves_fc2 = nn.Linear(head_hidden, 1)

    # ── Backbone ────────────────────────────────────────────────────────
    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        return out

    @staticmethod
    def _apply_legal_mask(logits: torch.Tensor, legal_mask: Optional[torch.Tensor]):
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

    # ── Forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,                     # (B, 13, 15, 15)
        rank_masks: torch.Tensor,            # (B, 4, 15, 15) — 1.0 at rank-k's cell
        legal_mask: Optional[torch.Tensor] = None,  # (B, 4) — rank-indexed
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cnn = self._backbone(x)              # (B, C, 15, 15)

        # Per-rank feature extraction via einsum (rank-indexed analogue of
        # V13.2's per-token extraction)
        per_rank = torch.einsum("bkij,bcij->bkc", rank_masks, cnn)   # (B, 4, C)

        p = F.relu(self.policy_fc1(per_rank))
        rank_logits = self.policy_fc2(p).squeeze(-1)                 # (B, 4)
        rank_logits = self._apply_legal_mask(rank_logits, legal_mask)
        policy = F.softmax(rank_logits, dim=1)

        pooled = F.adaptive_avg_pool2d(cnn, 1).flatten(1)            # (B, C)
        win_prob = torch.sigmoid(self.win_fc2(F.relu(self.win_fc1(pooled)))).squeeze(-1)
        moves    = F.softplus(self.moves_fc2(F.relu(self.moves_fc1(pooled)))).squeeze(-1)
        return policy, win_prob, moves

    def forward_policy_only(
        self,
        x: torch.Tensor,
        rank_masks: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-softmax legal-masked logits — used by RL/PPO sampler."""
        cnn = self._backbone(x)
        per_rank = torch.einsum("bkij,bcij->bkc", rank_masks, cnn)
        p = F.relu(self.policy_fc1(per_rank))
        rank_logits = self.policy_fc2(p).squeeze(-1)
        return self._apply_legal_mask(rank_logits, legal_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Helper: build (4, 15, 15) rank-masks from a state ────────────────────
def compute_rank_masks(state, v14_raw: Optional[np.ndarray] = None) -> np.ndarray:
    """For each canonical rank k, returns a 15×15 mask with 1.0 at the cell
    where rank-k's unique own-token position sits, 0.0 elsewhere.

    Ranks beyond the number of unique own-token positions are all-zero.

    The cell location for each non-home rank is read from V14_minimal's
    per-token planes (one plane per token-ID). For tokens at the same
    physical position, the cell is the same regardless of which token-ID
    we read — so picking any token-ID in the rank's group gives the same
    cell. For home ranks (rank position == -1), we use the canonical
    own-home cell (2, 2).

    Parameters
    ----------
    state : ludo_cpp.GameState
    v14_raw : optional precomputed V14_minimal encoding, shape (14, 15, 15).
        Pass this when you've already computed it elsewhere to avoid the
        extra C++ call.

    Returns
    -------
    masks : np.ndarray of shape (4, 15, 15), dtype float32.
    """
    if v14_raw is None:
        v14_raw = np.asarray(ludo_cpp.encode_state_v14_minimal(state), dtype=np.float32)

    cp = int(state.current_player)
    pp = state.player_positions[cp]    # (4,)
    rank_positions, rank_token_ids = state_to_rank_mapping(pp)

    out = np.zeros((MAX_RANK_SLOTS, 15, 15), dtype=np.float32)
    for k, (pos, token_ids) in enumerate(zip(rank_positions, rank_token_ids)):
        if k >= MAX_RANK_SLOTS:
            break
        if pos == HOME_POS:
            r, c = _OWN_HOME_RANK_CELL
            out[k, r, c] = 1.0
            continue
        # Read cell of any token-ID in this rank's group from V14 ch0..3
        # (all such tokens are at the same physical position, hence same cell).
        tok = token_ids[0]
        nz = np.argwhere(v14_raw[tok] > 0.5)
        if len(nz) == 0:
            # Shouldn't happen for non-home tokens; fall back to home cell.
            r, c = _OWN_HOME_RANK_CELL
            out[k, r, c] = 1.0
        else:
            r, c = int(nz[0, 0]), int(nz[0, 1])
            out[k, r, c] = 1.0
    return out


__all__ = [
    "V135Symmetric",
    "compute_rank_masks",
]
