"""V14_scalar — non-spatial DeepSets model with V12.2-equivalent feature set.

No CNN, no attention. Per-token MLPs + DeepSets pooling (sum + max).

Architecture (~250K params at default sizes):

  position_embedding: nn.Embedding(60, pos_emb_dim=32)         shared own/opp
  own_token_encoder:  Linear(40, 64) → ReLU → Linear(64, 64) → ReLU
  opp_token_encoder:  Linear(37, 64) → ReLU → Linear(64, 64) → ReLU
  global_encoder:     Linear(13, 64) → ReLU → Linear(64, 64) → ReLU

  pooling:
    own_pool = concat(sum(own_embs), max(own_embs))   # (B, 128)
    opp_pool = concat(sum(opp_embs), max(opp_embs))   # (B, 128)
    context  = concat(own_pool, opp_pool, global_emb) # (B, 320)

  trunk:
    Linear(320, 256) → ReLU → Linear(256, 256) → ReLU       # (B, 256)

  per-token policy head (shared weights, applied 4 times):
    inp[t] = concat(own_emb[t], context_minus_self[t], trunk) ≈ (B, 448)
    Linear(448, 64) → ReLU → Linear(64, 1)

  value head (single scalar from trunk):
    Linear(256, 64) → ReLU → Linear(64, 1) → sigmoid (win_prob in [0, 1])

Forward returns (policy, win_prob, moves_remaining=None) to mirror the
3-tuple shape of MinimalCNN14Aux / V12 / V12.2 — train_v12.py expects this.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Match encoder_v14_scalar.py constants
NUM_TOKENS = 4
NUM_OWN_FEATS = 8
NUM_OPP_FEATS = 5
NUM_GLOBALS = 13
NUM_POS_EMB = 60


class V14ScalarDeepSets(nn.Module):
    """V14_scalar: per-token MLP + DeepSets pooling. No CNN, no attention.

    Forward signature matches the V13.x family:
      forward(batch_dict, legal_mask=None) -> (policy, win_prob, moves_remaining)
    where moves_remaining is a zeros tensor (head not implemented; trainer
    sets the moves loss weight to 0 for this model).

    Args:
      pos_emb_dim:    embedding dim for token positions (shared own/opp). Default 32.
      token_hidden:   hidden width inside per-token encoders. Default 64.
      trunk_hidden:   trunk MLP width. Default 256.
      head_hidden:    policy/value head hidden width. Default 64.
    """

    def __init__(
        self,
        pos_emb_dim: int = 32,
        token_hidden: int = 64,
        trunk_hidden: int = 256,
        head_hidden: int = 64,
        num_pos: int = NUM_POS_EMB,
    ):
        super().__init__()
        self.pos_emb_dim = pos_emb_dim
        self.token_hidden = token_hidden
        self.trunk_hidden = trunk_hidden
        self.head_hidden = head_hidden

        # Shared position embedding (own & opp use the same lookup since
        # positions are absolute board cells with the same semantics).
        self.pos_emb = nn.Embedding(num_pos, pos_emb_dim)

        # Per-token encoders (shared across the 4 tokens within own/opp).
        own_in = pos_emb_dim + NUM_OWN_FEATS                  # 32 + 8 = 40
        opp_in = pos_emb_dim + NUM_OPP_FEATS                  # 32 + 5 = 37

        self.own_token_mlp = nn.Sequential(
            nn.Linear(own_in, token_hidden), nn.ReLU(),
            nn.Linear(token_hidden, token_hidden), nn.ReLU(),
        )
        self.opp_token_mlp = nn.Sequential(
            nn.Linear(opp_in, token_hidden), nn.ReLU(),
            nn.Linear(token_hidden, token_hidden), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(NUM_GLOBALS, token_hidden), nn.ReLU(),
            nn.Linear(token_hidden, token_hidden), nn.ReLU(),
        )

        # DeepSets pool: sum + max for each set → 2 * token_hidden per pool.
        # Context: own_pool || opp_pool || global_emb = 4 * token_hidden + token_hidden
        # = 5 * token_hidden = 320 at default.
        ctx_dim = 4 * token_hidden + token_hidden
        self.trunk = nn.Sequential(
            nn.Linear(ctx_dim, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )

        # Per-token policy head input dim:
        #   own_emb[t]            : token_hidden
        #   ctx_minus_self_pool[t]: 2 * token_hidden  (sum+max excluding self)
        #   opp_pool              : 2 * token_hidden
        #   global_emb            : token_hidden
        #   trunk_feat            : trunk_hidden
        policy_in = token_hidden + 2 * token_hidden + 2 * token_hidden + token_hidden + trunk_hidden
        self.policy_head = nn.Sequential(
            nn.Linear(policy_in, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Value head — single scalar from trunk, sigmoid for win_prob.
        self.value_head = nn.Sequential(
            nn.Linear(trunk_hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    # ────────── helpers ──────────
    def _encode_set(self, mlp, pos_idx, feats):
        """pos_idx: (B, 4) int64. feats: (B, 4, F) float32. Returns (B, 4, H)."""
        emb = self.pos_emb(pos_idx)                    # (B, 4, pos_emb_dim)
        x = torch.cat([emb, feats], dim=-1)            # (B, 4, pos_emb + F)
        # apply MLP per-token (shared weights)
        return mlp(x)                                  # (B, 4, H)

    @staticmethod
    def _set_pool(token_embs):
        """DeepSets sum+max pool. token_embs: (B, 4, H) → (B, 2H)."""
        s = token_embs.sum(dim=1)                      # (B, H)
        m = token_embs.amax(dim=1)                     # (B, H)
        return torch.cat([s, m], dim=-1)               # (B, 2H)

    @staticmethod
    def _leave_one_out_pool(token_embs):
        """For each token t, pool over the other 3 (sum+max). Returns (B, 4, 2H)."""
        B, N, H = token_embs.shape
        total_sum = token_embs.sum(dim=1, keepdim=True)            # (B, 1, H)
        # leave-one-out sum: total - this token's value
        loo_sum = total_sum - token_embs                            # (B, N, H)
        # leave-one-out max: per-token max over the other (N-1) tokens.
        # Cheap exact way: for each t, mask t out and take max.
        # We use a -inf-masked clone and take amax along dim=1.
        loo_max = torch.empty_like(token_embs)
        for t in range(N):
            mask = torch.ones(N, dtype=torch.bool, device=token_embs.device)
            mask[t] = False
            loo_max[:, t, :] = token_embs[:, mask, :].amax(dim=1)
        return torch.cat([loo_sum, loo_max], dim=-1)                # (B, N, 2H)

    @staticmethod
    def _apply_legal_mask(logits, legal_mask):
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

    # ────────── flat-tensor unpacking (RL trainer path) ──────────
    @staticmethod
    def _unpack_flat(flat: torch.Tensor) -> dict:
        """Unpack a flat tensor into the per-token + global dict.

        Accepts either:
          (B, FLAT_DIM, 1, 1) — what the RL trainer batches via the standard
                                CNN-shape pipeline
          (B, FLAT_DIM)       — pre-flattened
        """
        # Lazy-import sliced layout (avoid Python-side circular imports).
        from td_ludo.game.encoder_v14_scalar import (
            SLICE_OWN_POS, SLICE_OWN_FEAT, SLICE_OPP_POS, SLICE_OPP_FEAT,
            SLICE_GLOBALS, NUM_OWN_FEATS, NUM_OPP_FEATS,
        )
        if flat.dim() == 4:
            # (B, C, 1, 1) → (B, C)
            assert flat.size(-1) == 1 and flat.size(-2) == 1, (
                f"V14_scalar flat tensor must be (B, C, 1, 1); got {tuple(flat.shape)}"
            )
            flat = flat.squeeze(-1).squeeze(-1)
        B = flat.size(0)
        a, b = SLICE_OWN_POS;  own_pos = flat[:, a:b].long()
        a, b = SLICE_OWN_FEAT; own_feats = flat[:, a:b].reshape(B, NUM_TOKENS, NUM_OWN_FEATS)
        a, b = SLICE_OPP_POS;  opp_pos = flat[:, a:b].long()
        a, b = SLICE_OPP_FEAT; opp_feats = flat[:, a:b].reshape(B, NUM_TOKENS, NUM_OPP_FEATS)
        a, b = SLICE_GLOBALS;  globals_ = flat[:, a:b]
        return {
            "own_pos": own_pos, "own_features": own_feats,
            "opp_pos": opp_pos, "opp_features": opp_feats,
            "globals": globals_,
        }

    # ────────── forward ──────────
    def forward(self, batch, legal_mask: torch.Tensor | None = None):
        """Run a forward pass.

        Accepts EITHER:
          (a) batch dict with keys
                own_pos:      (B, 4) int64
                own_features: (B, 4, NUM_OWN_FEATS=8) float32
                opp_pos:      (B, 4) int64
                opp_features: (B, 4, NUM_OPP_FEATS=5) float32
                globals:      (B, NUM_GLOBALS=13) float32
          (b) flat tensor of shape (B, FLAT_DIM=73, 1, 1) or (B, 73)
              — RL trainer path. Unpacked internally to the dict above.

        Returns (policy, win_prob, moves_remaining) — moves_remaining is
        zeros (placeholder; head not implemented).
        """
        # If a tensor was passed (RL pipeline), unpack into the dict form.
        if isinstance(batch, torch.Tensor):
            batch = self._unpack_flat(batch)

        own_emb = self._encode_set(self.own_token_mlp, batch["own_pos"], batch["own_features"])
        opp_emb = self._encode_set(self.opp_token_mlp, batch["opp_pos"], batch["opp_features"])
        global_emb = self.global_mlp(batch["globals"])           # (B, H)

        own_pool = self._set_pool(own_emb)                       # (B, 2H)
        opp_pool = self._set_pool(opp_emb)                       # (B, 2H)
        context = torch.cat([own_pool, opp_pool, global_emb], dim=-1)  # (B, 5H)
        trunk_feat = self.trunk(context)                         # (B, trunk_hidden)

        # Per-token policy: for each own token t, build input from its own
        # embedding + leave-one-out pool over other own tokens + opp pool
        # + global emb + trunk feat. Shared head weights across t.
        own_loo = self._leave_one_out_pool(own_emb)              # (B, 4, 2H)
        B = own_emb.size(0)
        opp_pool_b = opp_pool.unsqueeze(1).expand(B, NUM_TOKENS, -1)        # (B, 4, 2H)
        global_b = global_emb.unsqueeze(1).expand(B, NUM_TOKENS, -1)        # (B, 4, H)
        trunk_b = trunk_feat.unsqueeze(1).expand(B, NUM_TOKENS, -1)         # (B, 4, trunk_hidden)
        policy_in = torch.cat([own_emb, own_loo, opp_pool_b, global_b, trunk_b], dim=-1)
        policy_logits = self.policy_head(policy_in).squeeze(-1)             # (B, 4)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        # Value head — sigmoid win_prob in [0, 1].
        win_prob = torch.sigmoid(self.value_head(trunk_feat)).squeeze(-1)   # (B,)

        # moves_remaining placeholder — trainer can ignore by setting weight 0.
        moves_remaining = torch.zeros_like(win_prob)

        return policy, win_prob, moves_remaining

    def forward_policy_only(self, batch, legal_mask=None):
        """Pre-softmax legal-masked policy logits. Mirrors V13.x interface.
        Accepts dict OR flat-tensor input (same dual interface as forward)."""
        if isinstance(batch, torch.Tensor):
            batch = self._unpack_flat(batch)
        own_emb = self._encode_set(self.own_token_mlp, batch["own_pos"], batch["own_features"])
        opp_emb = self._encode_set(self.opp_token_mlp, batch["opp_pos"], batch["opp_features"])
        global_emb = self.global_mlp(batch["globals"])

        own_pool = self._set_pool(own_emb)
        opp_pool = self._set_pool(opp_emb)
        context = torch.cat([own_pool, opp_pool, global_emb], dim=-1)
        trunk_feat = self.trunk(context)

        own_loo = self._leave_one_out_pool(own_emb)
        B = own_emb.size(0)
        opp_pool_b = opp_pool.unsqueeze(1).expand(B, NUM_TOKENS, -1)
        global_b = global_emb.unsqueeze(1).expand(B, NUM_TOKENS, -1)
        trunk_b = trunk_feat.unsqueeze(1).expand(B, NUM_TOKENS, -1)
        policy_in = torch.cat([own_emb, own_loo, opp_pool_b, global_b, trunk_b], dim=-1)
        policy_logits = self.policy_head(policy_in).squeeze(-1)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== V14_scalar DeepSets ===")
    model = V14ScalarDeepSets()
    print(f"Params: {model.count_parameters():,}")

    # Smoke-test forward with random batch
    B = 2
    batch = {
        "own_pos": torch.randint(0, NUM_POS_EMB, (B, NUM_TOKENS)),
        "own_features": torch.randn(B, NUM_TOKENS, NUM_OWN_FEATS),
        "opp_pos": torch.randint(0, NUM_POS_EMB, (B, NUM_TOKENS)),
        "opp_features": torch.randn(B, NUM_TOKENS, NUM_OPP_FEATS),
        "globals": torch.randn(B, NUM_GLOBALS),
    }
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    p, w, m = model(batch, mask)
    print(f"  policy: {tuple(p.shape)}, sums={p.sum(dim=1).tolist()}")
    print(f"  win_prob: {tuple(w.shape)}, vals={w.tolist()}")
    print(f"  moves_remaining: {tuple(m.shape)}")
