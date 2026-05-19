"""V13.5 production-pipeline adapter.

Wraps `V135Symmetric` so it can be plugged into `train_v12.py` /
`VectorACGamePlayer` without modifying the production infrastructure.

The production pipeline expects every model to expose:
  - `forward(x, legal_mask) -> (policy, win_prob, moves)` where `policy`
    is a (B, 4) softmax distribution INDEXED BY TOKEN-ID.
  - `forward_policy_only(x, legal_mask) -> logits` where `logits` is
    (B, 4) pre-softmax, also INDEXED BY TOKEN-ID.

V135Symmetric is rank-indexed and takes 3 inputs (x, rank_masks,
legal_mask). The production encoder packs everything we need into
the 21-channel input tensor; this adapter unpacks and bridges
between rank-indexed inner output and token-id-indexed outer output.

Token-id ↔ rank mapping
-----------------------
The packed encoder stores `token_to_rank: (B, 4)` as constant spatial
planes. From it we can:

    rank_legal[b, k] = OR over t such that token_to_rank[b, t] == k
                       of legal_mask_token_id[b, t]
                     = scatter_reduce(legal_mask_token_id, token_to_rank, max)

    token_logits[b, t] = rank_logits[b, token_to_rank[b, t]]
                       = gather(rank_logits, token_to_rank)

For tokens at the same rank the same logit is broadcast to all of
them — softmax + legal-mask + multinomial sampling then picks
uniformly among them, which is the correct invariance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from td_ludo.models.v13_5 import V135Symmetric
from td_ludo.game.encoder_v18_production import (
    V18_PROD_CHANNELS,
    V18_BASE_CHANNELS,
    RANK_MASK_CHANNELS,
    TOKEN_TO_RANK_CHANNELS,
    V18_BASE_SLICE,
    RANK_MASK_SLICE,
    unpack_token_to_rank,
)


class V135ProductionAdapter(nn.Module):
    """Token-id-indexed wrapper around V135Symmetric, for production RL pipeline."""

    def __init__(
        self,
        num_res_blocks: int = 10,
        num_channels: int = 128,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.inner = V135Symmetric(
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            in_channels=V18_BASE_CHANNELS,
            head_hidden=head_hidden,
        )

    # ── Helpers ─────────────────────────────────────────────────────────
    def _unpack(self, x: torch.Tensor):
        """Returns (v18_base, rank_masks, token_to_rank)."""
        v18 = x[:, V18_BASE_SLICE]                  # (B, 13, 15, 15)
        rank_masks = x[:, RANK_MASK_SLICE]          # (B, 4, 15, 15)
        token_to_rank = unpack_token_to_rank(x)     # (B, 4) int64
        # Defensive clamp (rank index is in 0..3, but unused slots default to 0
        # in the encoder which is fine because their rank_mask is zero anyway)
        token_to_rank = token_to_rank.clamp(0, RANK_MASK_CHANNELS - 1)
        return v18, rank_masks, token_to_rank

    def _build_rank_legal_mask(
        self,
        legal_mask_token_id: torch.Tensor,
        token_to_rank: torch.Tensor,
    ) -> torch.Tensor:
        """rank_legal[b, k] = max over tokens at rank k of legal_mask_token_id[b, t]."""
        B = legal_mask_token_id.size(0)
        rank_legal = torch.zeros(
            B, RANK_MASK_CHANNELS,
            dtype=legal_mask_token_id.dtype,
            device=legal_mask_token_id.device,
        )
        # scatter_reduce_ with reduce='amax' aggregates per-rank-group
        rank_legal.scatter_reduce_(
            dim=1,
            index=token_to_rank,
            src=legal_mask_token_id,
            reduce="amax",
            include_self=False,
        )
        return rank_legal

    @staticmethod
    def _apply_token_legal_mask(
        token_logits: torch.Tensor,
        legal_mask_token_id: torch.Tensor,
    ) -> torch.Tensor:
        """Set illegal-token logits to -inf. Falls back to all-zero if a row
        has no legal tokens (matches MinimalCNN14 behaviour for safety)."""
        all_illegal = (legal_mask_token_id.sum(dim=1, keepdim=True) == 0)
        masked = token_logits.masked_fill(~legal_mask_token_id.bool(), float("-inf"))
        if all_illegal.any():
            masked = torch.where(
                all_illegal.expand_as(masked),
                torch.zeros_like(masked),
                masked,
            )
        return masked

    # Class-level marker so trainer code can detect progress-head support
    # without inspecting the state dict.
    has_progress_head: bool = True

    # ── Production-pipeline interface (token-id-indexed) ────────────────
    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor):
        """x: (B, 21, 15, 15), legal_mask: (B, 4) [token-id-indexed].
        Returns (policy, win_prob, moves, progress) — policy and progress
        are token-id-indexed (broadcast via token_to_rank from rank-indexed
        inner output)."""
        v18, rank_masks, token_to_rank = self._unpack(x)
        rank_legal = self._build_rank_legal_mask(legal_mask, token_to_rank)

        # Inner forward — rank-indexed
        rank_policy, win_prob, moves, rank_progress = self.inner(
            v18, rank_masks, rank_legal,
        )

        # Broadcast rank_policy → token_policy via gather. Tokens at the same
        # rank get the same probability mass before legal-masking.
        token_policy = torch.gather(rank_policy, 1, token_to_rank)  # (B, 4)

        # Apply token-id legal mask + renormalize. Tokens at the same rank
        # that are both legal will share that rank's mass equally; tokens
        # at the same rank that are partially legal get 0 for the illegal
        # ones and full mass for the legal ones (then renormalized).
        token_policy = token_policy * legal_mask
        token_policy_sum = token_policy.sum(dim=1, keepdim=True)
        token_policy = token_policy / (token_policy_sum + 1e-8)

        # Broadcast rank_progress → token_progress (no legal-mask applied —
        # progress is a property of the token's position, not whether the
        # token can be moved this turn).
        token_progress = torch.gather(rank_progress, 1, token_to_rank)  # (B, 4)

        return token_policy, win_prob, moves, token_progress

    def forward_policy_only(self, x: torch.Tensor, legal_mask: torch.Tensor):
        """Pre-softmax legal-masked logits for sampler/PPO importance ratios.

        Strategy: run inner.forward_policy_only to get rank logits, gather
        to token logits, apply token-id legal mask. Tokens at the same rank
        share the same logit; softmax over masked logits gives them equal
        probability after illegal tokens are removed.
        """
        v18, rank_masks, token_to_rank = self._unpack(x)
        rank_legal = self._build_rank_legal_mask(legal_mask, token_to_rank)

        rank_logits = self.inner.forward_policy_only(v18, rank_masks, rank_legal)
        # rank_logits: (B, 4) — pre-softmax with -inf at illegal ranks.

        # Broadcast: token_logits[b, t] = rank_logits[b, token_to_rank[b, t]]
        token_logits = torch.gather(rank_logits, 1, token_to_rank)  # (B, 4)

        # Apply token-id legal mask (sets illegal token slots to -inf)
        return self._apply_token_legal_mask(token_logits, legal_mask)

    def count_parameters(self) -> int:
        return self.inner.count_parameters()

    # ── Checkpoint loading helpers ──────────────────────────────────────
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Override to transparently handle:
          1. bare V135Symmetric checkpoints (no `inner.` prefix)
          2. older V135 checkpoints that pre-date the progress-head addition
             (no `progress_fc1.*` / `progress_fc2.*` keys)

        For (1): rewrite keys to add `inner.` prefix.
        For (2): pre-pad the state_dict with the model's current (random init)
        progress-head weights, so a strict load still succeeds. The progress
        head will train from scratch in the next RL run.
        """
        # Strip torch.compile prefix if present
        if isinstance(state_dict, dict) and any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        if isinstance(state_dict, dict) and state_dict and not any(k.startswith("inner.") for k in state_dict):
            # Bare V135Symmetric format detected → prefix all keys with "inner."
            state_dict = {f"inner.{k}": v for k, v in state_dict.items()}

        # Pre-pad missing progress-head keys so strict loads still pass on
        # checkpoints saved before this head existed.
        if isinstance(state_dict, dict):
            current = self.state_dict()
            for k, v in current.items():
                if "progress_fc" in k and k not in state_dict:
                    state_dict[k] = v.clone()

        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            # Older PyTorch versions don't support `assign`
            return super().load_state_dict(state_dict, strict=strict)

    def load_v135_state_dict(self, sd: dict, strict: bool = True):
        """Convenience: explicitly load a bare V135Symmetric state_dict
        without going through the auto-detect path. Same effect as load_state_dict
        on a bare dict; kept for backwards compatibility."""
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        return self.inner.load_state_dict(sd, strict=strict)


def load_v135_into_adapter(
    adapter: V135ProductionAdapter,
    path: str,
    map_location=None,
    strict: bool = False,
):
    """Convenience: load a V135Symmetric or V135ProductionAdapter checkpoint
    into the given adapter. Auto-detects the prefix style."""
    sd = torch.load(path, map_location=map_location, weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    has_inner_prefix = any(k.startswith("inner.") for k in sd)
    if has_inner_prefix:
        # Saved by an adapter — load directly
        return adapter.load_state_dict(sd, strict=strict)
    # Saved as bare V135Symmetric — rewrite to inner.* and load
    return adapter.load_v135_state_dict(sd, strict=strict)


__all__ = ["V135ProductionAdapter", "load_v135_into_adapter"]
