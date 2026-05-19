"""V18 production encoder — packs V18 + rank info into a single tensor for the
production RL pipeline (`train_v12.py` / `VectorACGamePlayer`).

Why a separate "production" encoder?
The V13.5 architecture wants:
  - V18-symmetric input (13 channels)
  - Per-rank spatial mask channels (4)
  - A token→rank lookup (4 ints) so we can map between rank-indexed model
    output and token-id-indexed action selection

But the production pipeline (`v11.py:VectorACGamePlayer.forward_policy_only`)
expects a SINGLE encoder tensor input. So we pack everything into a
21-channel tensor:

    ch  0..12 : V18 base encoding         (own_count, opp_count, home_scalars,
                                          dice 6-one-hot, V11 statics)
    ch 13..16 : 4 rank-mask channels      rank k → 1.0 at the cell of rank k
    ch 17..20 : 4 token_to_rank planes    constant plane carrying token t's
                                          rank-index (as float, 0..3)

The V135ProductionAdapter unpacks these and runs the symmetric forward
internally, then converts rank-indexed output → token-id-indexed via
the packed token_to_rank lookup.
"""
from __future__ import annotations

import numpy as np
import td_ludo_cpp as ludo_cpp

from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric
from td_ludo.game.rank_mapping import (
    state_to_rank_mapping,
    MAX_RANK_SLOTS,
)
from td_ludo.models.v13_5 import compute_rank_masks


V18_PROD_CHANNELS = 21
V18_BASE_CHANNELS = 13      # ch 0..12
RANK_MASK_CHANNELS = 4      # ch 13..16
TOKEN_TO_RANK_CHANNELS = 4  # ch 17..20

V18_BASE_SLICE = slice(0, V18_BASE_CHANNELS)                        # 0..13
RANK_MASK_SLICE = slice(V18_BASE_CHANNELS, V18_BASE_CHANNELS + RANK_MASK_CHANNELS)  # 13..17
TOKEN_TO_RANK_SLICE = slice(
    V18_BASE_CHANNELS + RANK_MASK_CHANNELS,
    V18_BASE_CHANNELS + RANK_MASK_CHANNELS + TOKEN_TO_RANK_CHANNELS,
)  # 17..21


def encode_state_v18_production(state) -> np.ndarray:
    """Returns a (21, 15, 15) float32 array packing V18 + rank info."""
    # Reuse the V14 raw output for both the V18 base computation
    # (inside encode_state_v18_symmetric) and rank_masks (passed below)
    v14_raw = np.asarray(ludo_cpp.encode_state_v14_minimal(state), dtype=np.float32)

    # V18 base: 13 channels
    base = encode_state_v18_symmetric(state)  # (13, 15, 15)

    # Rank masks: 4 channels, 1.0 at the cell of each canonical rank
    rank_masks = compute_rank_masks(state, v14_raw=v14_raw)  # (4, 15, 15)

    # Token-to-rank lookup: for each token-id t in {0,1,2,3}, the canonical
    # rank index it belongs to (0=most-advanced, R-1=least-advanced/home).
    cp = int(state.current_player)
    pp = state.player_positions[cp]            # (4,)
    _, rank_token_ids = state_to_rank_mapping(pp)
    token_to_rank = np.zeros(4, dtype=np.int32)
    for k, tokens_in_rank in enumerate(rank_token_ids):
        if k >= MAX_RANK_SLOTS:
            break
        for t in tokens_in_rank:
            token_to_rank[t] = k

    # Broadcast each token's rank as a constant plane (so it survives the
    # spatial-tensor pipeline). Stored as float so the input dtype is uniform.
    trank_planes = np.zeros((4, 15, 15), dtype=np.float32)
    for t in range(4):
        trank_planes[t] = float(token_to_rank[t])

    out = np.empty((V18_PROD_CHANNELS, 15, 15), dtype=np.float32)
    out[V18_BASE_SLICE] = base
    out[RANK_MASK_SLICE] = rank_masks
    out[TOKEN_TO_RANK_SLICE] = trank_planes
    return out


def unpack_token_to_rank(x_batch):
    """Read the (B, 4) token_to_rank int tensor from a packed (B, 21, 15, 15) batch.

    Each token's rank-plane is constant across spatial dims, so reading any
    one cell (here [0, 0]) recovers the value. Returns int64 (long) tensor
    suitable for torch.gather.
    """
    import torch
    if isinstance(x_batch, torch.Tensor):
        # Read top-left corner of each plane in TOKEN_TO_RANK_SLICE
        ttr = x_batch[:, V18_BASE_CHANNELS + RANK_MASK_CHANNELS:
                          V18_BASE_CHANNELS + RANK_MASK_CHANNELS + TOKEN_TO_RANK_CHANNELS,
                       0, 0]  # (B, 4)
        return ttr.round().long()
    raise TypeError(f"unpack_token_to_rank expects torch.Tensor; got {type(x_batch)}")


__all__ = [
    "encode_state_v18_production",
    "unpack_token_to_rank",
    "V18_PROD_CHANNELS",
    "V18_BASE_SLICE",
    "RANK_MASK_SLICE",
    "TOKEN_TO_RANK_SLICE",
]
