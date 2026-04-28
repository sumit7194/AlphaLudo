"""
AlphaLudo V12 LEGACY — original (G=629K trained) architecture, kept for SL-data generation only.

Diagnosis from V11.1 gameplay analysis (Exp 20):
  V11.1's attention operates over 225 board CELLS — it can't directly reason
  about the 8 actual game pieces (4 own + 4 opp). Result: leader-greedy
  stacking, missed captures (model put 0% probability on capture-token in
  observed gameplay). Needs to attend over the entities, not the cells.

V12 design — same input encoding, attention moved to token entities:

  INPUT: (B, 28, 15, 15)  ← unchanged from V10/V11

  CNN backbone:
    4× ResBlock × 96ch  ← reuses V11.1 backbone (proven; SL works)

  Token-entity extraction (NEW):
    Use input channels 0-3 (own tokens, one-hot per token) and 17-20 (opp
    tokens, one-hot per token) as gather masks. For each token:
      feat = einsum('btij,bcij->btc', mask, cnn_features)
    yields (B, 8, num_channels) — the CNN feature at the cell where each
    of the 8 game pieces lives. No new model inputs needed; works because
    one-hot masks are 1.0 at exactly one cell each.

    Add learnable embeddings:
      - owner: mine (0) / opp (1)
      - token_idx: 0/1/2/3 within each owner

  Token attention:
    2× TransformerEncoderLayer over 8 tokens × 96 dim × 4 heads
    Attention map is only 8×8 — trivially small (~700× cheaper than V11).
    Each token can directly attend to all other 7 tokens.

  Combine:
    concat(GAP of CNN, mean of post-attn tokens) → 192 dim → heads

  Heads (same as V11):
    policy + win_prob (BCE) + moves_remaining (SmoothL1)

Why this fixes V11 weaknesses:
  - Leader-greedy stacking: token attention can compare all 4 own tokens
    directly, not just compare cells in a CNN feature map.
  - Capture-blindness: each own token sees opp tokens via attention; the
    relationship "opp at distance == dice → capturable" becomes learnable
    as a direct attention pattern between two tokens.
  - Idle-token blindness: per-token features include "where am I" implicitly
    (extracted from the cell-specific CNN feature), so attention can learn
    "this token has been at this position too long".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard CNN ResBlock — identical to V10/V11."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaLudoV12Legacy(nn.Module):
    """V12: CNN + token-entity attention over the 8 actual game pieces."""

    OWN_TOKEN_CHANNELS = (0, 1, 2, 3)      # input channels: my T0/T1/T2/T3 (one-hot per token)
    OPP_TOKEN_CHANNELS = (17, 18, 19, 20)  # input channels: opp T0/T1/T2/T3 (one-hot per token)

    def __init__(
        self,
        num_res_blocks: int = 4,
        num_channels: int = 96,
        num_attn_layers: int = 2,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
        in_channels: int = 28,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.num_attn_layers = num_attn_layers
        self.in_channels = in_channels

        # ---- Stem ----
        self.conv_input = nn.Conv2d(
            in_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # ---- CNN backbone ----
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ---- Token-entity embeddings ----
        # 2 owners × 4 token slots = 8 unique combinations, but we factor:
        #   owner_emb (2 × C) + token_idx_emb (4 × C)
        # so the model can learn "any-mine" vs "any-opp" semantics separately
        # from per-slot semantics.
        self.owner_emb = nn.Embedding(2, num_channels)        # 0=mine, 1=opp
        self.token_idx_emb = nn.Embedding(4, num_channels)    # T0..T3
        nn.init.trunc_normal_(self.owner_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.token_idx_emb.weight, std=0.02)

        # ---- Token-entity attention ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_channels,
            nhead=num_heads,
            dim_feedforward=num_channels * ffn_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=num_attn_layers
        )
        self.token_attn_norm = nn.LayerNorm(num_channels)

        # ---- Heads ----
        # Combine: GAP of CNN (96) + mean of attended tokens (96) = 192
        feat = num_channels * 2

        self.policy_fc1 = nn.Linear(feat, 64)
        self.policy_fc2 = nn.Linear(64, 4)

        self.win_fc1 = nn.Linear(feat, 64)
        self.win_fc2 = nn.Linear(64, 1)

        self.moves_fc1 = nn.Linear(feat, 64)
        self.moves_fc2 = nn.Linear(64, 1)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _cnn_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run stem + ResBlocks. Returns spatial features (B, C, H, W)."""
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        return out

    def _extract_token_features(
        self, x: torch.Tensor, cnn_features: torch.Tensor
    ) -> torch.Tensor:
        """Gather per-token features from CNN at the cells where each token sits.

        x:            (B, in_channels, H, W) — input tensor (we read ch 0-3 & 17-20 as masks)
        cnn_features: (B, num_channels, H, W) — backbone output

        Returns: (B, 8, num_channels) — 4 own + 4 opp token features.

        How: input ch i (i in {0..3, 17..20}) is one-hot at the cell where
        token i sits (own) or opp-token i sits. einsum('btij,bcij->btc') with
        the mask multiplies the CNN feature by the one-hot at each cell and
        sums spatially — yielding the cell's feature vector at the token's
        position. For tokens in BASE, the one-hot is at a base-area cell.
        For SCORED tokens (pos 99), the one-hot is at the home center (7,7).
        """
        # (B, 4, H, W) for each set
        own_mask = x[:, list(self.OWN_TOKEN_CHANNELS)]
        opp_mask = x[:, list(self.OPP_TOKEN_CHANNELS)]

        # Gather. Result: (B, 4, num_channels) for each set.
        # einsum: for each batch b, token t, channel c: sum over (i,j) of mask[b,t,i,j] * features[b,c,i,j]
        own_features = torch.einsum("btij,bcij->btc", own_mask, cnn_features)
        opp_features = torch.einsum("btij,bcij->btc", opp_mask, cnn_features)

        # Add owner + token-idx embeddings
        # token_idx: 0..3
        token_idx = torch.arange(4, device=x.device)
        token_idx_e = self.token_idx_emb(token_idx)  # (4, C)
        own_owner_e = self.owner_emb(torch.zeros(1, dtype=torch.long, device=x.device))  # (1, C)
        opp_owner_e = self.owner_emb(torch.ones(1, dtype=torch.long, device=x.device))   # (1, C)

        # Broadcast: (B, 4, C) + (4, C) + (1, C) → (B, 4, C)
        own_features = own_features + token_idx_e.unsqueeze(0) + own_owner_e.unsqueeze(0)
        opp_features = opp_features + token_idx_e.unsqueeze(0) + opp_owner_e.unsqueeze(0)

        # Concatenate into (B, 8, C)
        return torch.cat([own_features, opp_features], dim=1)

    def _apply_legal_mask(
        self, policy_logits: torch.Tensor, legal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if legal_mask is None:
            return policy_logits
        all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
        policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float("-inf"))
        if all_illegal.any():
            policy_logits = torch.where(
                all_illegal.expand_as(policy_logits),
                torch.zeros_like(policy_logits),
                policy_logits,
            )
        return policy_logits

    def _build_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run full backbone (CNN + token attention) and return combined features."""
        cnn_features = self._cnn_backbone(x)  # (B, C, H, W)

        # Per-token features at game-piece cells
        tokens = self._extract_token_features(x, cnn_features)  # (B, 8, C)
        tokens = self.token_attention(tokens)
        tokens = self.token_attn_norm(tokens)
        token_summary = tokens.mean(dim=1)  # (B, C)

        # Spatial summary
        cnn_gap = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)  # (B, C)

        return torch.cat([cnn_gap, token_summary], dim=1)  # (B, 2C)

    # ------------------------------------------------------------------
    # Forward signatures — match V10/V11 (drop-in for trainer_v10).
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None):
        features = self._build_features(x)

        p = F.relu(self.policy_fc1(features))
        policy_logits = self._apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        w = F.relu(self.win_fc1(features))
        win_prob = torch.sigmoid(self.win_fc2(w)).squeeze(-1)

        m = F.relu(self.moves_fc1(features))
        moves_remaining = F.softplus(self.moves_fc2(m)).squeeze(-1)

        return policy, win_prob, moves_remaining

    def forward_policy_only(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self._build_features(x)
        p = F.relu(self.policy_fc1(features))
        return self._apply_legal_mask(self.policy_fc2(p), legal_mask)

    def forward_with_features(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ):
        features = self._build_features(x)
        p = F.relu(self.policy_fc1(features))
        policy_logits = self._apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        w = F.relu(self.win_fc1(features))
        win_prob = torch.sigmoid(self.win_fc2(w)).squeeze(-1)

        m = F.relu(self.moves_fc1(features))
        moves_remaining = F.softplus(self.moves_fc2(m)).squeeze(-1)

        return policy, win_prob, moves_remaining, features

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== V12: CNN + Token-Entity Attention ===")
    model = AlphaLudoV12()
    print(f"params: {model.count_parameters():,}")

    # Smoke test forward
    x = torch.randn(2, 28, 15, 15)
    # Make tokens 0-3 and 17-20 actual one-hot (random cells per token)
    for batch in range(2):
        for ch in [0, 1, 2, 3, 17, 18, 19, 20]:
            x[batch, ch] = 0
            r, c = torch.randint(0, 15, (2,)).tolist()
            x[batch, ch, r, c] = 1.0
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    policy, win_prob, moves = model(x, mask)
    print(f"\npolicy:           {tuple(policy.shape)} sums={policy.sum(dim=1).tolist()}")
    print(f"win_prob:         {tuple(win_prob.shape)} values={win_prob.tolist()}")
    print(f"moves_remaining:  {tuple(moves.shape)} values={moves.tolist()}")

    # Smoke test gradient flow
    loss = policy.sum() + win_prob.sum() + moves.sum()
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"\ngrad-flow OK: {grad_count} params received gradients")

    # Param breakdown
    cnn_params = sum(p.numel() for n, p in model.named_parameters()
                     if any(s in n for s in ['conv_input', 'bn_input', 'res_blocks']))
    attn_params = sum(p.numel() for n, p in model.named_parameters()
                      if any(s in n for s in ['token_attention', 'token_attn_norm', 'owner_emb', 'token_idx_emb']))
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if any(s in n for s in ['policy_fc', 'win_fc', 'moves_fc']))
    print(f"\nParam breakdown:")
    print(f"  CNN backbone:                 {cnn_params:>9,}")
    print(f"  Token-entity attention:       {attn_params:>9,}")
    print(f"  Heads (input dim 192):        {head_params:>9,}")
    print(f"  Total:                        {model.count_parameters():>9,}")
