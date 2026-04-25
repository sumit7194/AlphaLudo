"""
AlphaLudo V11 — ResTNet hybrid (CNN backbone + Transformer attention).

Drop-in successor to V10. Identical input encoding (28 channels) and identical
3-head output (policy + win_prob + moves_remaining), so trainer_v10.py and the
existing C++ encoder both work unchanged.

The change is the backbone:
  V10:                  V11:
    stem                  stem
    6× ResBlock(96)       4× ResBlock(96)         ← 2 ResBlocks freed for attention
    GAP                   reshape (B,96,15,15) → (B,225,96)
    heads                 + learned 2D pos-embed
                          2× TransformerEncoderLayer(96d, 4 heads, FFN 384, pre-norm)
                          reshape back → GAP
                          heads

Motivation (research synthesis 2026-04-25 + V6 mech interp):
- Pure CNN can't reason about long-range dependencies in one forward pass —
  each ResBlock grows receptive field by 2; reasoning across a 15×15 board
  about distant tokens (capture chains, multi-token threats, bonus-turn
  2-step plans) requires many CNN layers and dilutes signal.
- Self-attention reads all 225 cells in one pass.
- ResTNet (Wu et al, IJCAI 2025, arXiv 2410.05347) showed +6-7pp WR over
  pure ResNet on 9×9 Go, 19×19 Go, 19×19 Hex. Notably fixed long-range
  ladder reading (59% → 80%) — the kind of reasoning V6 mech interp said
  our CNN family lacks.

Param count: ~1.35M (V10 = 1.04M, +30% for attention layers + pos-embed).
Token count for attention: 15×15 = 225 (manageable; 225² = 50K attention map).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard 2-conv ResBlock with BatchNorm. Identical to V10's."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaLudoV11(nn.Module):
    """V11: CNN backbone + Transformer attention + 3-head output.

    Args:
        num_res_blocks: ResBlocks in the CNN backbone (default 4, vs V10's 6).
        num_channels: Channel width / token dim (default 96).
        num_attn_layers: Transformer encoder layers (default 2).
        num_heads: Attention heads per layer (default 4).
        ffn_ratio: FFN inner dim multiplier (default 4 → ffn_dim=384 at 96ch).
        dropout: Dropout in attention + FFN. Default 0.0 (PPO-safe).
        in_channels: Input channels (default 28, matches encode_state_v10).
        board_size: Spatial dim per side (default 15 for AlphaLudo).
    """

    def __init__(
        self,
        num_res_blocks: int = 4,
        num_channels: int = 96,
        num_attn_layers: int = 2,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
        in_channels: int = 28,
        board_size: int = 15,
        attn_dim: int | None = None,
    ):
        """
        Args:
            attn_dim: Inner dimension of the transformer (Q/K/V/FFN width).
                If None (default), uses num_channels — V11 original behavior.
                If set smaller (e.g. 64 with num_channels=96), adds Linear
                projection in/out around the transformer stack.
                This lets us shrink attention memory + compute (V11.1) while
                keeping CNN at full width and all 225 tokens at exact
                spatial precision.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.num_attn_layers = num_attn_layers
        self.in_channels = in_channels
        self.board_size = board_size
        self.num_tokens = board_size * board_size  # 225 for 15×15
        self.attn_dim = attn_dim if attn_dim is not None else num_channels

        # ---- Stem: 28ch input → num_channels features ----
        self.conv_input = nn.Conv2d(
            in_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # ---- CNN backbone ----
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ---- Optional attention dim projection (V11.1: 96 → 64 → 96) ----
        # If attn_dim != num_channels, project into smaller dim before
        # attention and back after. nn.Identity for V11 default (no-op).
        if self.attn_dim != num_channels:
            self.attn_in_proj = nn.Linear(num_channels, self.attn_dim, bias=False)
            self.attn_out_proj = nn.Linear(self.attn_dim, num_channels, bias=False)
        else:
            self.attn_in_proj = nn.Identity()
            self.attn_out_proj = nn.Identity()

        # ---- Learned 2D positional embedding (one per board cell) ----
        # Shape matches attention dim: (1, num_tokens, attn_dim).
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.attn_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ---- Transformer encoder ----
        # Pre-norm (norm_first=True): more stable training, especially with PPO.
        # batch_first=True: easier shape management.
        # Activation gelu: modern transformer default.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attn_dim,
            nhead=num_heads,
            dim_feedforward=self.attn_dim * ffn_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_attn_layers
        )
        # Final layer-norm after the transformer stack — common pre-norm pattern.
        self.attn_out_norm = nn.LayerNorm(self.attn_dim)

        feat = num_channels

        # ---- Heads: identical to V10 (policy + win_prob + moves_remaining) ----
        self.policy_fc1 = nn.Linear(feat, 48)
        self.policy_fc2 = nn.Linear(48, 4)

        self.win_fc1 = nn.Linear(feat, 48)
        self.win_fc2 = nn.Linear(48, 1)

        self.moves_fc1 = nn.Linear(feat, 48)
        self.moves_fc2 = nn.Linear(48, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run stem + ResBlocks + Transformer + GAP. Returns (B, num_channels)."""
        # Stem
        out = F.relu(self.bn_input(self.conv_input(x)))

        # CNN backbone
        for block in self.res_blocks:
            out = block(out)
        # out: (B, num_channels, H, W) — full spatial resolution preserved
        conv_features = out  # save for residual skip

        # Spatial → tokens for attention. (B, C, H, W) → (B, H*W, C)
        B, C, H, W = out.shape
        tokens = out.flatten(2).transpose(1, 2)  # (B, H*W, num_channels)

        # Project down to attention dim if needed (V11.1: 96 → 64)
        tokens = self.attn_in_proj(tokens)  # (B, H*W, attn_dim)

        # Add positional embedding (matches attn_dim)
        tokens = tokens + self.pos_embedding

        # Transformer (pre-norm internally; layers carry residual)
        tokens = self.transformer(tokens)
        tokens = self.attn_out_norm(tokens)

        # Project back up to num_channels for skip connection (V11.1: 64 → 96)
        tokens = self.attn_out_proj(tokens)  # (B, H*W, num_channels)

        # Tokens → spatial. (B, H*W, C) → (B, C, H, W)
        out = tokens.transpose(1, 2).view(B, C, H, W)

        # Residual: combine attention output with CNN features so attention
        # is "additive refinement" not "replacement". Helps init stability
        # and ensures CNN tactical info isn't lost if attention learns badly.
        out = conv_features + out

        # Global average pool → (B, C)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(start_dim=1)

    def _apply_legal_mask(
        self, policy_logits: torch.Tensor, legal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Mask illegal moves to -inf; uniform fallback if all illegal."""
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

    # ------------------------------------------------------------------
    # Forward signatures — identical to V10 (drop-in compatible with trainer_v10).
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None):
        """Full forward: returns (policy, win_prob, moves_remaining).

        Shapes:
          x: (B, 28, 15, 15)
          legal_mask: (B, 4) optional binary mask
          policy: (B, 4) softmax probabilities
          win_prob: (B,) sigmoid in [0, 1]
          moves_remaining: (B,) softplus >= 0
        """
        features = self._backbone_features(x)

        # Policy
        p = F.relu(self.policy_fc1(features))
        policy_logits = self._apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        # Win prob (sigmoid, BCE-trained)
        w = F.relu(self.win_fc1(features))
        win_prob = torch.sigmoid(self.win_fc2(w)).squeeze(-1)

        # Moves remaining (softplus, SmoothL1-trained)
        m = F.relu(self.moves_fc1(features))
        moves_remaining = F.softplus(self.moves_fc2(m)).squeeze(-1)

        return policy, win_prob, moves_remaining

    def forward_policy_only(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fast inference path — returns masked policy logits only."""
        features = self._backbone_features(x)
        p = F.relu(self.policy_fc1(features))
        return self._apply_legal_mask(self.policy_fc2(p), legal_mask)

    def forward_with_features(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ):
        """Return all 3 heads + post-attention GAP features (for probing)."""
        features = self._backbone_features(x)
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

    # ------------------------------------------------------------------
    # V10 → V11 weight transfer (warm start from V10 checkpoint)
    # ------------------------------------------------------------------
    def load_v10_backbone(self, v10_state_dict: dict, strict_heads: bool = True):
        """Copy V10's stem + as many ResBlocks as fit + heads.

        V11 has fewer ResBlocks (4) than V10 (6); we copy the first 4 and
        leave the transformer / pos_embedding random-initialized.

        Returns dict {copied: list, skipped: list, missing: list}.
        """
        own = self.state_dict()
        copied, skipped = [], []

        for k, v in v10_state_dict.items():
            if k not in own:
                skipped.append(k)
                continue
            if own[k].shape == v.shape:
                own[k] = v.clone()
                copied.append(k)
            else:
                skipped.append(f"{k} (shape mismatch: own={list(own[k].shape)} vs ckpt={list(v.shape)})")

        self.load_state_dict(own, strict=False)

        # Identify what was missing (transformer, pos_embed, etc.)
        missing = [k for k in own if k not in v10_state_dict]
        return {"copied": copied, "skipped": skipped, "missing": missing}


if __name__ == "__main__":
    print("=== V11 (default — for comparison) ===")
    model = AlphaLudoV11()
    print(f"V11 params: {model.count_parameters():,}")
    print(f"  num_res_blocks: {model.num_res_blocks}")
    print(f"  num_attn_layers: {model.num_attn_layers}")
    print(f"  num_channels: {model.num_channels}")
    print(f"  attn_dim: {model.attn_dim}")
    print(f"  in_channels: {model.in_channels}")
    print(f"  num_tokens (attention): {model.num_tokens}")

    print()
    print("=== V11.1 (reduced attention) ===")
    model_v11_1 = AlphaLudoV11(
        num_res_blocks=4, num_channels=96,
        num_attn_layers=1, num_heads=2,
        ffn_ratio=4, attn_dim=64,
    )
    print(f"V11.1 params: {model_v11_1.count_parameters():,}")
    print(f"  num_attn_layers: {model_v11_1.num_attn_layers}")
    print(f"  num_channels: {model_v11_1.num_channels}")
    print(f"  attn_dim: {model_v11_1.attn_dim}")
    model = model_v11_1  # smoke test the reduced variant

    # Smoke test forward
    x = torch.randn(2, 28, 15, 15)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    policy, win_prob, moves = model(x, mask)
    print(f"\npolicy:           {tuple(policy.shape)} sums={policy.sum(dim=1).tolist()}")
    print(f"win_prob:         {tuple(win_prob.shape)} values={win_prob.tolist()}")
    print(f"moves_remaining:  {tuple(moves.shape)} values={moves.tolist()}")

    # Smoke test policy_only
    logits = model.forward_policy_only(x, mask)
    print(f"\nforward_policy_only logits: {tuple(logits.shape)}")

    # Smoke test gradient flow
    loss = policy.sum() + win_prob.sum() + moves.sum() + logits.sum()
    loss.backward()
    grads = {n: p.grad.abs().mean().item() for n, p in model.named_parameters() if p.grad is not None}
    print(f"\ngrad-flow OK: {len(grads)} params have gradients")
    print(f"  pos_embedding grad mean: {grads.get('pos_embedding', 0):.6f}")
    print(f"  transformer.layers.0.self_attn.in_proj_weight grad: "
          f"{grads.get('transformer.layers.0.self_attn.in_proj_weight', 0):.6f}")
    print(f"  conv_input.weight grad: {grads.get('conv_input.weight', 0):.6f}")
    print(f"  policy_fc2.weight grad: {grads.get('policy_fc2.weight', 0):.6f}")

    # Param count breakdown
    backbone_params = sum(p.numel() for n, p in model.named_parameters()
                          if any(s in n for s in ['conv_input', 'bn_input', 'res_blocks']))
    attn_params = sum(p.numel() for n, p in model.named_parameters()
                      if any(s in n for s in ['transformer', 'pos_embedding', 'attn_out_norm']))
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if any(s in n for s in ['policy_fc', 'win_fc', 'moves_fc']))
    print(f"\nParam breakdown:")
    print(f"  CNN (stem+resblocks): {backbone_params:,}")
    print(f"  Attention (pos+trans+norm): {attn_params:,}")
    print(f"  Heads: {head_params:,}")
    print(f"  Total: {model.count_parameters():,}")
